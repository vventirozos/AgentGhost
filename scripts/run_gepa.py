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
import math
import os
import shutil
import calendar as _calendar_mod
import json as _json_mod
import time as _time_mod
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
from ghost_agent.optim import ab_eval
from ghost_agent.optim import gate_contract  # noqa: E402
from ghost_agent.optim.signatures import SIGNATURES  # noqa: E402
from ghost_agent.optim.trainset import (  # noqa: E402
    build_trainset,
    per_origin_selection,
    real_only_gate,
    split_public_private,
    split_train_eval,
)



# ── THE METRIC, AT MODULE LEVEL ─────────────────────────────────────
# ⚠ These were NESTED inside `main()`, so no other tool could reach
# them. `scripts/recheck_gepa_incumbent.py` tried, got AttributeError
# on every example, and `ab_eval._run_one`'s broad except turned that
# into `passed=False` for BOTH arms — an instrument that could only
# report zero, which then printed a confident verdict about the live
# artifact. Lifted so there is ONE definition of 'did this prompt
# win', shared by the optimizer's gate and by any re-check.
def _overlap(want: str, got: str) -> float:
    """Token F1 — NOT recall.

    Recall (`|w & g| / |w|`) makes VERBOSITY the optimum. Re-measured
    2026-08-04 against a real 87-token gold plan from the live corpus
    (`planning.decompose`, n=96 plan targets, median 35 distinct tokens):

        candidate                 recall     F1
        terse correct subset       0.333    0.500
        gold + 300 filler tokens   1.000    0.367
        that soup vs UNRELATED gold 0.250   0.047

    Recall ranks the soup ABOVE the correct answer and still gives it
    0.250 against a gold it never addressed; F1 inverts both. A hidden
    holdout defends against memorising items — it cannot defend against
    a metric whose optimum generalises.

    The cost of F1 is length sensitivity in the OTHER direction: a
    perfectly-recalling answer much longer than the gold falls under the
    0.3 pass bar. That is survivable here only because the gold is a
    PLAN (n=96, median 35 distinct tokens, p90 58 — re-measured
    2026-08-04; an earlier revision said p90 61) rather than a whole
    final reply — which is exactly why `build_trainset` must keep
    yielding plan targets (see the per-field kind filter there). If that
    ever collapses to the `final_response` fallback, revisit this bar.

    ⚠ THIS METRIC CHANGE INVALIDATES THE PROMOTED planning.decompose
    ARTIFACT AS A MEASURED WIN. Re-run 2026-08-04 on the same hash-stable
    28-example private tier, both arms at temp 0 / no-think: under RECALL
    the promoted artifact scores 0.857 vs the seed's 0.429 (+0.429 —
    reproducing the 2026-07-29 promotion, journal 0.45 -> 0.80); under F1
    it scores 0.071 vs 0.500 (-0.429). Its outputs run a median 111
    distinct tokens against a 32-token median gold. Neither metric
    measures plan QUALITY, so this is a correctness-of-record finding,
    not proof the artifact is bad — but the promotion decision does not
    reproduce under the objective this function now implements. The
    read-site is dark — and until 2026-08-07 was UNREACHABLE in any
    configuration: no `--use-planning` flag existed anywhere, only
    tests set the attribute (§4L Lens-C MAJOR-3). The flag is real
    now; the artifact stays unapplied until the operator boots with
    it.
    """
    w = set(re.findall(r"[a-z0-9_]+", want.lower()))
    g = set(re.findall(r"[a-z0-9_]+", got.lower()))
    if not w or not g:
        return 0.0
    hits = len(w & g)
    if not hits:
        return 0.0
    recall = hits / len(w)
    precision = hits / len(g)
    return 2.0 * precision * recall / (precision + recall)


def _gold_field(fields_obj, sig=None) -> str:
    """WHICH output field the gold actually carries.

    ⚠ THIS EXISTS BECAUSE THE METRIC WAS ASYMMETRIC AND THE ASYMMETRY SET
    A VERDICT'S SIGN. `_expected_target` returns the FIRST non-empty
    output field — `plan` for `planning.decompose` — while the prediction
    side joined EVERY output field (`plan` + `rationale`). `build_trainset`
    never stamps `rationale` on a gold, so a two-field prediction was
    scored against a one-field target and token-F1 precision was capped
    by construction: the more a prompt invested in the ungraded field, the
    worse it scored.

    Measured 2026-08-24 on the retired `planning.decompose` artifact,
    n=31 private tier:

        arm                     recall  precision      F1   pass@0.3
        hand-written seed        0.294      0.339   0.285      15/31
        artifact, as measured    0.366      0.223   0.258      11/31
        artifact, `plan` only    0.309      0.400   0.315      14/31

    Its ONLY deficit was precision, and its `### rationale` section adds a
    median 26 distinct tokens against a median 30-token gold — it nearly
    doubles precision's denominator. Grade its plan alone and the sign of
    the F1 delta FLIPS (+0.029), and its recall is better at p=0.005.

    The fix is symmetry: score the same field the gold carries.
    """
    for f in list(getattr(sig, "outputs", ()) or ()) + ["final_response"]:
        v = (fields_obj.get(f, "") if isinstance(fields_obj, dict)
             else getattr(fields_obj, f, ""))
        if str(v or "").strip():
            return f
    return ""


def _prediction_for(pred, field: str, sig=None) -> str:
    """The prediction text to score, matched to the gold's field.

    Falls back to the joined output fields, then to the raw completion,
    when the structured field is absent — a candidate that ignored the
    format should be scored on what it did emit, not on "".
    """
    if field:
        v = str(getattr(pred, field, "") or "").strip()
        if v:
            return v
    joined = " ".join(str(getattr(pred, f, "") or "")
                      for f in (getattr(sig, "outputs", ()) or ())).strip()
    return joined or str(pred or "")


def _significance_floor() -> int:
    """Delegates to `ab_eval.significance_floor` — one derivation shared by
    both GEPA runners and the miner, so the instrument cannot drift from
    the gate it reports."""
    return ab_eval.significance_floor()


#: The token-F1 bar a prediction must clear. ONE literal, because
#: `scripts/recheck_gepa_incumbent.py` carried a second copy of it — two
#: definitions of "did this prompt win" is how two answers to the same
#: question come to disagree.
_PASS_BAR = 0.3

#: A free-text completion has no attributes, so the A/B runner has to
#: find the gold's field in the raw text. The artifact instructs
#: `### plan` / `### rationale`; the seed emits neither, and then the
#: whole reply IS the plan — which is why this returns "" rather than
#: guessing, and the caller falls back to the full text.
_SECTION_RX = r"(?:^|\n)\s*#{1,4}\s*%s\b[^\n]*\n(.*?)(?=\n\s*#{1,4}\s|\Z)"


def _section_of(text: str, field: str) -> str:
    """The `### <field>` section of a free-text completion, or ""."""
    import re as _re
    m = _re.search(_SECTION_RX % _re.escape(field), str(text or ""),
                   _re.S | _re.I)
    return m.group(1).strip() if m else ""


def _expected_target(fields_obj, sig=None) -> str:
    """First non-empty signature-output field on the gold (falling back
    to final_response). `fields_obj` is a dspy.Example after
    `_to_dspy_examples` — attribute access — or a raw dict.

    ⚠ `sig` is a PARAMETER now. Nested inside `main()` it closed over the
    signature, which is what made it unreachable from any other tool —
    and a re-check that could not call it produced `passed=False` for
    both arms and a confident wrong verdict.
    """
    outputs = list(getattr(sig, "outputs", ()) or ())
    for f in outputs + ["final_response"]:
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
    # §4CV: mined verifiable-reward examples. OPT-IN — a trainset that
    # silently changed shape would invalidate every comparison against a
    # previous run.
    parser.add_argument("--allow-seed-loss", action="store_true",
                        help="promote even when the candidate LOSES to the "
                             "hand-written seed instruction. The gate "
                             "normally refuses that, because comparing only "
                             "against the previous artifact lets the chain "
                             "ratchet away from the baseline unnoticed.")
    parser.add_argument("--mined-bank", default=None, metavar="NAME",
                        help="Add §4CV mined failure-environment examples "
                             "from system/optim/mined_envs/NAME.jsonl. They "
                             "carry origin='bench', so `real_only_gate` keeps "
                             "them out of the PRIVATE ship tier: they may "
                             "TEACH, never GRADE. Their examples are scored "
                             "by their own EXECUTABLE ORACLE instead of token "
                             "overlap.")
    parser.add_argument("--ab-gate", action="store_true", default=False,
                        help="(deprecated no-op — the A/B gate is on by default)")
    parser.add_argument("--no-ab-gate", action="store_true", default=False,
                        help="Skip the A/B gate and adopt the tuned prompt UNVERIFIED. Use only when there is no eval split to compare on.")
    parser.add_argument("--ab-min-delta", type=float, default=0.02,
                        help="Minimum eval pass-rate improvement for --ab-gate to ship the candidate. Default 0.02.")
    parser.add_argument(
        "--min-promotion-age-days", type=float, default=7.0,
        help="Refuse to re-promote a signature whose live artifact is "
             "younger than this many days (0 disables). Each run draws a "
             "fresh candidate, so frequent runs against a slowly-growing "
             "holdout are repeated draws at the same gate.")
    parser.add_argument(
        "--allow-insignificant-ship", action="store_true",
        help="Promote a candidate that clears --ab-min-delta but whose "
             "discordant pairs do not reach McNemar p<=SHIP_ALPHA. The "
             "default refuses, because `delta > margin` alone promoted on "
             "a net swing of TWO examples out of a 50-example holdout — "
             "25-40%% of the time under the null. Use this when the margin "
             "is large and the holdout is simply too small to reach "
             "significance (a 4-0 sweep sits at p=0.0625); it is RECORDED "
             "in the artifact's gate block so the call can be audited "
             "later.")
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

    # §4BF 1c (admissibility: gepa_trainset = bench_feature): bench solves
    # join the corpus — a PASSED bench final_response was accepted by the
    # bank's mechanical oracle, and build_trainset stamps origin="bench" on
    # every example it yields. Missing bench root → empty, silently.
    bench_trajs = []
    try:
        from ghost_agent.core.admissibility import iter_bench_trajectories
        bench_trajs = list(iter_bench_trajectories("gepa_trainset"))
        if bench_trajs:
            print(f"admitting {len(bench_trajs)} bench trajectories "
                  f"(origin=bench) alongside {len(trajectories)} real ones")
            trajectories = trajectories + bench_trajs
    except Exception as e:  # noqa: BLE001 — bench is additive
        print(f"bench corpus skipped: {e}", file=sys.stderr)

    # No cap yet — the signature-target filter below must see the whole
    # corpus first (plan-bearing trajectories are ~1 in 4 of PASSED; capping
    # first would truncate away most of the usable examples).
    examples = build_trainset(
        trajectories,
        signature_name=sig.name,
        max_examples=None,
    )

    # §4CV — mined failure environments. Each carries an EXECUTABLE
    # checker, so `_metric` below scores it by RUNNING the checker rather
    # than by token overlap against a recorded reply. That is the whole
    # point of the mining: the 191 FAILED turns in the corpus contribute
    # nothing to `build_trainset` (a failure has no gold answer to overlap
    # against), and overlap rewards LOOKING like the recorded answer.
    _mined_by_request = {}
    if args.mined_bank:
        try:
            from ghost_agent.optim.env_mining import (
                read_staging, signature_can_use_mined, trainset_from_items,
            )
            # ⚠ REFUSE SIGNATURES THE JOIN KEY CANNOT SURVIVE. `_metric`
            # keys on `gold.user_request`, and `_to_dspy_examples` copies
            # only the signature's DECLARED inputs — so for
            # `tool_selection.pick` / `reflection.critique` the challenge
            # text is discarded and the golds arrive all-empty, scoring
            # 0.0 in BOTH arms. Round 2 measured exactly that. Adding
            # noise that looks like data is worse than adding nothing.
            if not signature_can_use_mined(sig):
                raise RuntimeError(
                    f"{sig.name} has no 'user_request' input, so a mined "
                    f"challenge cannot reach the metric — mined examples "
                    f"would arrive empty and score 0.0 in both arms")
            _rows = read_staging(args.mined_bank)   # $GHOST_HOME
            # Stamp the reference into THIS signature's own output
            # fields, or the `keyed` filter below drops every one of them
            # (round 2: 0 of 1 survived on the live corpus).
            _mined = trainset_from_items(_rows, sig.name,
                                         outputs=sorted(sig.outputs))
            _by_id = {str(r.get("item_id") or ""): r for r in _rows}
            for _ex in _mined:
                _row = _by_id.get(_ex.source_trajectory_id)
                if _row:
                    _mined_by_request[_ex.inputs["user_request"]] = _row
            _dropped = len(_rows) - len(_mined)
            _artifact = sum(1 for r in _rows
                            if str(r.get("graded_on") or "artifact")
                            != "final_response")
            print(f"§4CV: {len(_mined)} mined example(s) with executable "
                  f"oracles from '{args.mined_bank}'"
                  + (f" ({_dropped} row(s) dropped: {_artifact} "
                     f"artifact-graded — a GEPA rollout produces TEXT and "
                     f"cannot write a solution file — and "
                     f"{_dropped - _artifact} malformed)" if _dropped else ""))
            if not _mined and _rows:
                print("⚠ every mined row was dropped — nothing from this "
                      "bank reaches the optimizer", file=sys.stderr)
            examples = examples + _mined
        except Exception as e:  # noqa: BLE001 — mined examples are additive
            print(f"mined bank skipped: {e}", file=sys.stderr)

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
    #
    # ⚠ KNOWN COMPOSITION CONSEQUENCE, measured 2026-08-04 on the live corpus
    # (1488 trajectories): `planning_output` is populated on reflection
    # trajectories and NOWHERE ELSE (157 of 157), so for `planning.decompose`
    # this filter keeps 96 examples that are 100% reflection-sourced and
    # discards all 410 clean PASSED user_request examples. §4J's headline was
    # "19% of the GEPA train set was reflection plans"; after the per-field
    # fix in `build_trainset` the poisoned FIELD (a reflection's diagnosis
    # final_response) is gone, but the surviving planning corpus is entirely
    # revised plans from failed attempts. That is defensible — a revised plan
    # IS a plan — and it is NOT what the journal line describes. Do not read
    # a planning.decompose run as trained on ordinary user turns.
    keyed = [e for e in examples
             if any((e.expected_output or {}).get(f) for f in sig.outputs)]
    # ⚠ SAY IT IF IT HAPPENS ANYWAY. Round 2 found `--mined-bank` losing
    # 100% of its examples to this filter with no message at all — the
    # run looked normal and the oracle simply never fired. A silent
    # zeroing of the thing the flag exists to add is the failure mode
    # this whole section is about.
    if args.mined_bank:
        _mined_kept = sum(1 for e in keyed
                          if getattr(e, "origin", "") == "bench"
                          and e.inputs.get("user_request") in _mined_by_request)
        if _mined_by_request and not _mined_kept:
            print(f"⚠ §4CV: ALL {len(_mined_by_request)} mined example(s) "
                  f"were dropped by the {sorted(sig.outputs)} target filter "
                  f"— the oracle cannot fire. Aborting rather than running "
                  f"a trainset that silently lost the flag's whole effect.",
                  file=sys.stderr)
            return 2
    if len(keyed) >= 20:
        if len(keyed) < len(examples):
            print(f"filtered to {len(keyed)}/{len(examples)} examples with a "
                  f"{sorted(sig.outputs)} target")
        examples = keyed
    elif keyed:
        print(f"only {len(keyed)} examples carry a signature-output target "
              f"(<20) — keeping all {len(examples)}; metric falls back to "
              f"final_response overlap")
    else:
        # `elif keyed:` is ALSO false at zero, so the total-collapse case
        # printed nothing and the run silently optimized + ship-gated the
        # signature against whole final replies instead of its own outputs.
        # Zero is the loudest case, not the quietest.
        print(f"⚠ NONE of {len(examples)} examples carry a "
              f"{sorted(sig.outputs)} target — every example will be graded "
              f"by final_response overlap, INCLUDING the private ship gate. "
              f"That is a different objective from {sig.name}; check the "
              f"corpus before trusting a promotion.", file=sys.stderr)
    # §4BF 1c (R1 review): per-origin selection instead of head truncation
    # — see optim/trainset.py:per_origin_selection for the rules.
    _real_ex, _bench_ex = per_origin_selection(
        examples, max_examples=args.max_examples)
    examples = _real_ex + _bench_ex
    if _bench_ex or bench_trajs:
        print(f"corpus after per-origin selection: {len(_real_ex)} real + "
              f"{len(_bench_ex)} bench examples (equal-mass cap, "
              f"newest bench first)")

    # PUBLIC/PRIVATE first: the private tier is hash-assigned per trajectory
    # and is the ONLY thing the A/B ship-gate judges on. The optimizer —
    # including its internal train/val split below — works exclusively on
    # the public tier. Judging the ship decision on data the optimizer
    # could see is how proxy-gamed prompts get promoted.
    public_set, private_set = split_public_private(examples, private_pct=args.private_pct)
    # §4BF 1c (R1 review CRIT): the PRIVATE ship-gate is REAL-ONLY — bench
    # may teach, it may never grade. See optim/trainset.py:real_only_gate.
    public_set, private_set, _n_moved = real_only_gate(public_set, private_set)
    if _n_moved:
        print(f"evicted {_n_moved} bench examples from the PRIVATE gate "
              f"tier (real-only gate); public tier now "
              f"{sum(1 for e in public_set if getattr(e, 'origin', '') != 'bench')} real + "
              f"{sum(1 for e in public_set if getattr(e, 'origin', '') == 'bench')} bench "
              f"(equal-mass cap enforced)")
    train_set, eval_set = split_train_eval(public_set, eval_fraction=args.eval_fraction)
    print(f"{len(train_set)} train / {len(eval_set)} val (public) / "
          f"{len(private_set)} PRIVATE holdout examples for {sig.name}")

    # ── RESOLUTION CHECK, *BEFORE* the expensive part ─────────────────
    # The gate's smallest possible non-zero delta is 1/n. Shipping on a
    # `min_delta` FINER than the metric can resolve means a single flipped
    # example decides the run — measured on the sibling verifier gate, whose
    # 6-case non-refute arm has a 0.083 step against a 0.02 threshold, which
    # is the arithmetic cause of the journal's "+-0.08 private-gate noise".
    #
    # This depends ONLY on `len(private_set)` and `--ab-min-delta`, both
    # known here — it used to sit after `run_gepa(...)`, so a run that could
    # never ship burned the whole optimization first and then exited 1. The
    # sibling optimize_verifier.py gates before `gepa.optimize` and says
    # "REFUSING TO RUN"; this now matches.
    # ⚠ TWO preconditions, and the operator must be told the BINDING one.
    #
    # Resolution (`1/n <= min_delta`) says the metric can REPRESENT the
    # threshold. It says nothing about power. A one-sided exact McNemar
    # cannot reach SHIP_ALPHA=0.05 with fewer than 5 discordant pairs
    # (4-0 is p=0.0625), so a holdout below that cannot ship whatever the
    # candidate does — the whole optimization would be paid for and then
    # refused.
    #
    # ⚠ An earlier version of this comment claimed the floor catches
    # `--max-examples 10 --private-pct 60 --ab-min-delta 0.2`. It does
    # not: that gives n=5, and 5 < 5 is False while 0.2 > 0.2 is also
    # False — neither guard fires, and after the one-sided switch that run
    # SHIPS a 5-0 sweep correctly. At the 0.02 default the resolution
    # requirement (50) dwarfs the floor (5), so the floor only binds for
    # `--ab-min-delta >= 0.25`. It is a backstop for coarse-margin runs,
    # not a safety net at defaults. Reporting the two separately let the
    # operator satisfy the weaker one, re-run, and only then learn the
    # real requirement, so they are combined into one number.
    # ⚠ `--ab-min-delta` must be a usable margin BEFORE anything divides
    # by it. `0` gave an uncaught ZeroDivisionError out of `main()` (the
    # older, division-free form refused cleanly); `>= 1` is arithmetically
    # unsatisfiable — `delta > 1.0` cannot happen — so the run would pay
    # for the whole optimization and then refuse everything.
    # ⚠ RATE CAP — refuse a re-promotion that arrives before the last one
    # could possibly have been judged. Checked HERE, with the resolution and
    # significance pre-flights, so a capped run costs nothing rather than
    # buying the whole optimization first.
    #
    # The point is not tidiness. Each GEPA run draws a fresh candidate, so
    # repeated runs against a near-static holdout are repeated draws at the
    # gate: at the measured accrual (0.62 private examples/day) a weekly
    # cadence re-decides on essentially the same evidence, and even the
    # §4CY gate's 1-3% per-run false-promotion rate compounds to ~0.5-0.8
    # over 52 draws. Spacing promotions is what converts that back into a
    # per-run number.
    if args.min_promotion_age_days > 0 and output_path.exists():
        try:
            _stamp = ((_json_mod.loads(output_path.read_text()).get("gate")
                       or {}).get("promoted_utc") or "")
            _age = None
            if _stamp:
                _t = _time_mod.strptime(_stamp, "%Y-%m-%dT%H:%M:%SZ")
                _age = (_time_mod.time() - _calendar_mod.timegm(_t)) / 86400.0
        except Exception:  # noqa: BLE001 — an unreadable stamp must not
            _age = None   # block a run; it is treated as "age unknown".
        # ⚠ A NEGATIVE AGE IS A CLOCK, NOT A RECENT PROMOTION. A stamp in
        # the future (skew, a hand-edited artifact, a restored backup) is
        # "less than the cap" on a naive comparison, so the signature would
        # refuse every run until wall-clock caught up — an unbounded outage
        # from a one-character bug in someone else's file. Treated as
        # "age unknown", the same as a missing or corrupt stamp.
        if _age is not None and _age < 0:
            _age = None
        if _age is not None and _age < args.min_promotion_age_days:
            print(f"REFUSING TO RUN: the live artifact was promoted "
                  f"{_age:.1f} days ago and --min-promotion-age-days is "
                  f"{args.min_promotion_age_days}. Each run is a fresh draw "
                  f"at the gate, so re-promoting before the last one can be "
                  f"judged turns one decision into many. Wait, or pass "
                  f"--min-promotion-age-days 0 to override deliberately.",
                  file=sys.stderr)
            # ⚠ 2, NOT 1. This file already uses 2 for "cannot run"
            # (no corpus, no trajectories, too few examples) and used
            # 1 — "the gate rejected the candidate" — for five states
            # in which nothing was measured. That is the collision
            # §4DA rounds 11/13/15 carved codes out for in the two
            # judges, left whole in the gate the rule was ported FROM.
            return 2

    # ⚠ A LOWER BOUND, not just >0. `1e-320` passes `0 < x` and then
    # `math.ceil(1.0 / x)` raises an uncaught OverflowError out of
    # `main()` — the same failure, from the same expression, as the
    # ZeroDivisionError this guard was added to close.
    if not 1e-6 <= args.ab_min_delta < 1:
        print(f"REFUSING TO RUN: --ab-min-delta {args.ab_min_delta} is not "
              f"a usable margin. It must be >=1e-6 (a bar of 0 admits any "
              f"non-zero swing; anything smaller cannot be resolved by a "
              f"holdout of any size this project can build) and <1 (no "
              f"pass-rate delta can exceed 1.0, so nothing could ship).",
              file=sys.stderr)
        # ⚠ 2, NOT 1 — see the re-draw guard above. An unusable margin is
        # a broken invocation, not a verdict about the candidate.
        return 2

    _min_discordant = _significance_floor()
    _resolution_need = math.ceil(1.0 / args.ab_min_delta)
    _need = max(_min_discordant, _resolution_need)
    if len(private_set) < _need:
        # ⚠ SAY THE TRUE THING PER REASON, AND OFFER THE REMEDY THAT WORKS.
        # A single "No candidate could ship" was FALSE in 81% of refusals:
        # it is true only when the tier is below the significance floor.
        # Below the RESOLUTION requirement a candidate can still ship — a
        # 5-0 sweep at n=45 is delta +0.111 at p=0.031 — the run is refused
        # because one flipped example would decide it, which is policy, not
        # impossibility. The remedy hint was also exactly inverted: raising
        # the margin fixes a resolution refusal and can NEVER fix a floor
        # refusal, and it was offered in precisely the wrong branch.
        _below_floor = len(private_set) < _min_discordant
        # `<`, not `<=`: at n == _resolution_need the margin IS
        # resolvable, so adding the resolution reason there states a
        # second, false cause for a refusal the floor alone produced.
        _below_res = len(private_set) < _resolution_need
        _why = []
        if _below_floor:
            _why.append(f"even a perfect sweep needs {_min_discordant} "
                        f"discordant pairs to reach "
                        f"p<={ab_eval.SHIP_ALPHA}, so NO candidate could "
                        f"ship at any margin")
        if _below_res:
            # ⚠ `:.3f` TRUNCATED THE STEP INTO THE BAR. At n=49 vs 50 and
            # --ab-min-delta 0.02 the clause printed "a smallest step of
            # 0.020 cannot resolve --ab-min-delta 0.02" for the refusal and
            # nothing for the run that proceeded — two identically-rendering
            # numbers, opposite decisions, and the near-miss tier is exactly
            # the one an operator sees while the corpus grows.
            # ⚠ THE EXACT FRACTION, not only its decimal. `:.3f` collided
            # with the bar for a whole class of tiers; `:.6g` narrowed that
            # to 191 of 400 (a bar chosen AS the 6-figure rounding of 1/n,
            # e.g. n=7 -> "step of 0.142857 cannot resolve 0.142857"). No
            # fixed-precision decimal closes it, because the bar can always
            # be written at that precision. `1/n` cannot collide with a
            # decimal, so the reader can always see which is larger.
            _why.append(f"a smallest step of 1/{len(private_set)} "
                        f"({1.0 / max(1, len(private_set)):.6g}) cannot "
                        f"resolve --ab-min-delta {args.ab_min_delta}, so a "
                        f"single flipped example would decide the run")
        # ⚠ VERIFY THE STRING THE OPERATOR WILL TYPE, NOT THE FLOAT.
        # Printed as `{1/n:.3f}` the offer rounds DOWN and re-triggers the
        # identical refusal — a fixed point at 172 of the 396 tier sizes
        # in 5..400, including the live 31 and the harness's 45.
        #
        # The first fix asserted `ceil(1.0 / _offer) <= n` on the computed
        # float. That is TRUE BY CONSTRUCTION for `_offer = 1/n`, so it
        # guarded a proxy and could not see the bug it was written for
        # (`guard-a-proxy-not-the-thing`). What matters is the rounded
        # value the message actually renders, so that is what is checked.
        _OFFER_DP = 3
        _offer = math.ceil(10 ** _OFFER_DP / max(1, len(private_set))) / 10 ** _OFFER_DP
        _typed = float(f"{_offer:.{_OFFER_DP}f}")
        # ⚠ AND AN EMPTY TIER HAS NO OFFER TO MAKE. Round 11 guarded the
        # DIVISION with `max(1, …)` and left this assertion reading the
        # real length, so `--private-pct 0` traded a ZeroDivisionError
        # for an AssertionError and the "the PRIVATE holdout is empty"
        # message 350 lines below stayed unreachable. Brute-forced
        # n=0..5000: n=0 is the only failing tier — exactly the case the
        # fix was written for.
        assert not private_set or (
            _typed > 0 and math.ceil(1.0 / _typed) <= len(private_set)), (
            f"the offer as PRINTED ({_typed}) still refuses at "
            f"n={len(private_set)}")
        if not private_set:
            print("REFUSING TO RUN: the PRIVATE holdout is EMPTY (0 "
                  "examples). With --private-pct 0 there is nothing to "
                  "gate on, so no candidate could ever ship. Raise "
                  "--private-pct, or pass --no-ab-gate to promote "
                  "deliberately without a measurement.", file=sys.stderr)
            # ⚠ 2, NOT 1 — an empty holdout measured nothing.
            return 2
        _fix = (f"Collect at least {_need} private examples"
                + ("." if _below_floor else
                   f", or raise --ab-min-delta to at least "
                   f"{_offer:.{_OFFER_DP}f} (which does NOT lower the "
                   f"{_min_discordant}-pair significance floor)."))
        print(f"REFUSING TO RUN: {len(private_set)} private examples is "
              f"not enough — {'; and '.join(_why)}. {_fix}", file=sys.stderr)
        # ⚠ 2, NOT 1 — a tier that cannot resolve the margin measured
        # nothing; the candidate was never scored.
        return 2


    # Build LLM client + metric
    from ghost_agent.core.llm import LLMClient
    llm_client = LLMClient(args.upstream_url)

    # Ignition metric: graded token recall of the expected target inside the
    # prediction's declared output fields. Deterministic, zero extra LLM
    # calls, and GRADED — GEPA needs a gradient, and the old binary
    # substring check scored ~everything 0. Replaced by real benches
    # (verify_bench / replay fixtures) in §4F Phase 2.
    _oracle_stats = {"scored": 0, "unrunnable": 0, "passed": 0}

    def _metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
        # ⚠ SYMMETRIC. This used to join EVERY output field against a
        # gold that carries ONE — see `_gold_field`. Score what the gold
        # is, or the metric measures format compliance.
        _gf = _gold_field(gold, sig)
        got = _prediction_for(pred, _gf, sig)

        # §4CV: a mined example carries its own EXECUTABLE checker, and a
        # verifiable reward beats a string-similarity proxy — that is the
        # entire reason the mining exists. Keyed on the request text,
        # which is what dspy carries through into `gold`.
        _req = str(getattr(gold, "user_request", "") or "")
        _row = _mined_by_request.get(_req)
        if _row is not None:
            from ghost_agent.optim.env_mining import (
                _extract_answer, oracle_score,
            )
            # ⚠ NOT THE CONCATENATION. `got` joins EVERY output field,
            # and `planning.decompose` — the only joinable signature —
            # emits (plan, rationale). Round 3 measured the consequence:
            # a bare `42` scores 1.0, but the realistic
            # `"42 Because 6 times 7 is 42."` scores 0.0, so a mined
            # example becomes a CONSTANT-ZERO column for essentially
            # every real rollout. "A metric that can only ever reject" —
            # the failure these docstrings cite three times — and
            # `_report_oracle_use` could not see it, because the oracle
            # DOES fire. The round-2 fix verified the join and never
            # that the joined score could be non-zero.
            #
            # Scored per FIELD, through the same `_extract_answer` the
            # probe uses, so the metric and the gate that admitted the
            # item ask the same question of the same string. Best field
            # wins: the challenge says "reply with only the value", and a
            # candidate that put the value in one field answered it.
            _cands = [str(getattr(pred, f, "") or "") for f in sig.outputs]
            _cands.append(got)
            _scores = [oracle_score(_row, _extract_answer(c))
                       for c in _cands if c.strip()]
            _real = [x for x in _scores if x is not None]
            if _real:
                _oracle_stats["scored"] += 1
                if max(_real) >= 1.0:
                    _oracle_stats["passed"] += 1
                return max(_real)
            # ⚠ NOT 0.0. `oracle_score` returns None when the checker
            # could not be RUN, and scoring an infrastructure failure as
            # "the candidate was wrong" optimises against noise. Falling
            # back to overlap keeps the example in play on a signal that
            # at least means something; the count is reported after the
            # run so a silent all-fallback run cannot pass as an oracle
            # run.
            _oracle_stats["unrunnable"] += 1

        want = _expected_target(gold, sig)
        if not want:
            return 0.0
        return _overlap(want, got)

    def _report_oracle_use():
        """Say whether the oracle metric actually FIRED.

        A mined bank whose checker never ran is a trainset scored by
        overlap wearing a verifiable-reward label — the built-but-unwired
        failure, one level in. `§4CS` is the standing lesson: give the
        number, not the intention.
        """
        if not args.mined_bank:
            return
        sc, un = _oracle_stats["scored"], _oracle_stats["unrunnable"]
        ps = _oracle_stats["passed"]
        if not sc and not un:
            print("⚠ §4CV: the mined bank was loaded but its oracle NEVER "
                  "FIRED — every example was scored by token overlap. "
                  "Check that the mined requests reach the metric.",
                  file=sys.stderr)
            return
        print(f"§4CV: the executable oracle scored {sc} rollout(s), "
              f"{ps} of them PASSING"
              + (f"; {un} fell back to overlap because the checker "
                 f"could not be run" if un else ""))
        # ⚠ FIRING IS NOT THE SAME AS DISCRIMINATING. Round 3: the
        # oracle fired on every rollout and returned 0.0 on all of them,
        # because the signature emits structured fields and the mined
        # challenges ask for a bare value. A constant-zero column carries
        # no gradient and the run looks normal — so the number that
        # matters is how many PASSED, and zero says so out loud.
        if sc and not ps:
            print(f"⚠ §4CV: the oracle fired {sc} time(s) and NOTHING "
                  f"PASSED. A constant-zero column gives the optimizer no "
                  f"gradient — the mined items are not answerable in this "
                  f"signature's output shape. Treat this run's mined "
                  f"contribution as ABSENT, not as evidence.",
                  file=sys.stderr)

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

    # ⚠ INITIALISED BEFORE `_promote_staging` is even DEFINED. The seed
    # arm is computed near the END of `main()`, and the closure below
    # reads it for provenance — so the `--no-ab-gate` path, which
    # promotes long before that line runs, raised NameError. A
    # provenance field that crashes the one path adopting a prompt
    # UNVERIFIED is the worst possible place to put one.
    _seed_cmp = None
    #: Mutable so the promotion stamp (a closure defined above) can see an
    #: override decided below it.
    _seed_override = [False]
    #: Same, for a deliberate ship-side override of the significance bar.
    _ship_override = [False]

    def _promote_staging():
        # ⚠ BUILD AND VALIDATE THE STAMP BEFORE ANYTHING MOVES. The stamp
        # used to be constructed AFTER `os.replace`, so both of this
        # round's guards misreported the world when they fired (final
        # verification pass, finding 1): a `validate_gate_record` refusal
        # exited 2 — "nothing changed, re-run when stable" — with the
        # CANDIDATE already live and unstamped, and a `build_seed_arm`
        # refusal (the delta-identity check, the one that catches a rate
        # swap) fired inside the stamp's I/O swallow and exited 0 —
        # PROMOTED, gate block absent. The sibling gate validates before
        # promoting; now this one does too: a contract breach leaves the
        # incumbent untouched and the staging file on disk for
        # post-mortem.
        _stamp = _build_gate_stamp()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Keep the incumbent. `os.replace` onto the live path used to be the
        # ONLY copy operation, so a candidate that beat a stale baseline
        # silently destroyed a better artifact (measured: the live
        # planning.decompose scored 0.80 on its own private gate; a 0.50
        # candidate beating the hard-coded 200-char baseline by +0.05 would
        # have replaced it, unrecoverably).
        if output_path.exists():
            backup = output_path.with_suffix(output_path.suffix + ".prev")
            try:
                shutil.copy2(output_path, backup)
                print(f"incumbent backed up to {backup}")
            except OSError as e:
                print(f"WARNING: could not back up incumbent ({e}) — "
                      "promotion aborted", file=sys.stderr)
                raise
        os.replace(staging_path, output_path)
        # ⚠ Stamp the GATE IDENTITY into the promoted artifact (§4L
        # D-MAJOR-1 follow-through): the staging writer emits the bare
        # schema, so even a legitimately-gated promotion produced an
        # artifact the loader must warn about ("predates the gate
        # schema"). The stamp is what makes "this artifact won under
        # THIS metric" checkable later — the missing provenance that
        # let the recall-metric planning.decompose survive the F1
        # change unchallenged.
        try:
            _art = _json_mod.loads(output_path.read_text(encoding="utf-8"))
            _art["gate_arm"], _art["gate"] = _stamp
            # ⚠ STAGED + os.replace, LIKE THE PROMOTION TWENTY LINES
            # ABOVE. This was a truncate-then-write on the LIVE path —
            # the very shape §4DA round 3 fixed in the sibling promoter,
            # in a function that had already done
            # `os.replace(staging_path, output_path)` and then re-opened
            # the file to stamp it. A torn write leaves invalid JSON,
            # `loader._CACHE[sig]` caches `None` for the life of the
            # process (repairing the file does NOT recover it), and
            # `gepa_live_check` could then derive no sha and acted on a
            # POOLED arm — driven, that turned KEEP p=0.6122 into
            # REVERT p=0.0065 and retired a healthy artifact.
            _stamp_tmp = output_path.with_suffix(
                output_path.suffix + ".stamp")
            _stamp_tmp.write_text(_json_mod.dumps(_art, indent=1),
                                  encoding="utf-8")
            os.replace(_stamp_tmp, output_path)
        except Exception as _se:  # noqa: BLE001 — stamp must not unship
            print(f"WARNING: gate stamp failed ({_se}) — artifact "
                  f"promoted without provenance", file=sys.stderr)

    def _build_gate_stamp():
        """(gate_arm, gate) — built and VALIDATED before anything moves.

        A ValueError from ANY schema check here (`validate_gate_record`,
        `build_seed_arm`'s identity and exclusivity rules) becomes a loud
        exit 2 with the incumbent untouched — never a live-but-unstamped
        candidate, and never the I/O swallow's "promoted without
        provenance".
        """
        # `cmp` exists only on the GATED path (R2 review: the
        # --no-ab-gate promotion hit a NameError here, was swallowed,
        # and every ungated promotion shipped with the "promoted
        # without provenance" warning instead of an honest stamp).
        try:
            _cmp = cmp
        except NameError:
            _cmp = None
        try:
            if _cmp is not None:
                # ⚠ VERSIONED. `gate_arm` exists so "this artifact won
                # under THIS metric" is checkable, and §4DA changed the
                # DENOMINATOR of delta and both pass rates (all examples
                # -> examples that reached a verdict in both arms) while
                # leaving this string byte-identical to the one on
                # `planning.decompose.json.retired-4cw`, decided under the
                # old meaning. Two artifacts whose gate_arm matches are
                # supposed to be comparable; these were not. The project's
                # own `gepa-promoted-artifact-invalidation` rule ("re-score
                # the incumbent when the metric or gate changes") had no
                # way to fire.
                _gate_arm = ("token-F1 A/B, private holdout "
                                    f"[{ab_eval.GATE_METRIC_VERSION}]")
                _gate = {
                    "metric": "token_f1_overlap>=0.3",
                    # ⚠ BUILT FROM THE SHARED SCHEMA. This block and the
                    # tool-description gate's had drifted into two
                    # different shapes for the same arm; one builder now
                    # owns both. `delta` is seed-minus-candidate, the
                    # direction the veto fires in. Without the exclusion
                    # fields the block once recorded a FABRICATED perfect
                    # tie for a totally-outaged seed arm: 0.0/0.0, 0 wins,
                    # 0 ties, with 45 exclusions nowhere in the file.
                    "seed_arm": (None if _seed_cmp is None else
                                 gate_contract.build_seed_arm(
                                     seed_pass_rate=(
                                         _seed_cmp.baseline_pass_rate),
                                     candidate_pass_rate=(
                                         _seed_cmp.candidate_pass_rate),
                                     seed_minus_candidate_delta=(
                                         -_seed_cmp.delta),
                                     seed_minus_candidate_raw_delta=(
                                         -_seed_cmp.raw_delta),
                                     n_usable_pairs=(
                                         len(private_set)
                                         - _seed_cmp.transport_excluded),
                                     transport_excluded=(
                                         _seed_cmp.transport_excluded),
                                     seed_wins=_seed_cmp.baseline_wins,
                                     candidate_wins=(
                                         _seed_cmp.candidate_wins),
                                     p_value=_seed_cmp.p_value,
                                     vetoed=bool(_seed_loses),
                                     overridden=bool(_seed_override[0]))),
                    "n_private": len(private_set),
                    # ⚠ THE FIELDS `recheck_gepa_incumbent` WAS BUILT TO
                    # READ. §4DA rounds 4-5 taught that reader to report
                    # the usable pair count and the exclusion causes, and
                    # only the SIBLING optimizer wrote them — so the
                    # warning was structurally unreachable for every
                    # `run_gepa` artifact, i.e. for `planning.decompose`,
                    # the signature that reader's docstring is about.
                    "n_usable_pairs": (len(private_set)
                                       - _cmp.transport_excluded),
                    "transport_excluded": _cmp.transport_excluded,
                    "outage_excluded": _cmp.transport_excluded,
                    "corpus_gap_excluded": 0,
                    # ⚠ AND SAY THAT THE SPLIT ABOVE IS NOT MEASURED HERE.
                    # `recheck_gepa_incumbent.py` prints these two back
                    # verbatim as "(N transport outage, 0 no recorded
                    # payload)", and `ab_eval._run_one` marks ANY runner
                    # exception `UNREACHED` — a metric bug, a malformed
                    # example and a per-example timeout all land in the
                    # first bucket, which is the one the reader calls
                    # re-runnable. This gate replays LIVE, so there is no
                    # recorded payload to lose and no predicate to
                    # separate the causes with; the sibling gate replays
                    # RECORDINGS and counts both by predicate. Same key
                    # names as round 7 established — one more field
                    # saying how much they are worth.
                    "exclusion_cause_distinguished": False,
                    "incumbent_pass_rate": round(_cmp.baseline_pass_rate, 4),
                    "candidate_pass_rate": round(_cmp.candidate_pass_rate, 4),
                    "delta": round(_cmp.delta, 4),
                    # ⚠ THE RAW RATES TOO, NOT JUST THE RAW MARGIN. This
                    # block recorded `raw_delta` with neither rate, so a
                    # `planning.decompose` artifact could not reconstruct
                    # the all-rows comparison at all — and the sibling
                    # gate's raw pair lives at TOP level as
                    # `private_incumbent`/`private_candidate`, which this
                    # runner has no equivalent of. Recording them here
                    # duplicates nothing in this file and makes the gate
                    # block self-contained in both runners.
                    "raw_incumbent_pass_rate": round(
                        _cmp.raw_baseline_pass_rate, 4),
                    "raw_candidate_pass_rate": round(
                        _cmp.raw_candidate_pass_rate, 4),
                    "raw_delta": round(_cmp.raw_delta, 4),
                    "min_delta": args.ab_min_delta,
                    # The evidence behind the delta, not just the delta.
                    # A promotion whose record cannot answer "how many
                    # examples actually moved?" is unauditable — which is
                    # how the §4CW artifact served every planner turn for
                    # weeks on a win nobody could reproduce.
                    "p_value": (None if _cmp.p_value is None
                                else round(_cmp.p_value, 6)),
                    "ship_alpha": ab_eval.SHIP_ALPHA,
                    "discordant_pairs": (_cmp.baseline_wins
                                         + _cmp.candidate_wins),
                    "candidate_wins": _cmp.candidate_wins,
                    "incumbent_wins": _cmp.baseline_wins,
                    "significance_overridden": _ship_override[0],
                    "promoted_utc": __import__("time").strftime(
                        "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                }
            else:
                _gate_arm = "UNGATED (--no-ab-gate)"
                _gate = {
                    "metric": "none — adopted unverified",
                    "promoted_utc": __import__("time").strftime(
                        "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                }
            gate_contract.validate_gate_record(
                _gate, writer="scripts/run_gepa.py")
        except ValueError as _ve:
            # ⚠ SystemExit(<string>) exits 1 — "the gate rejected the
            # candidate" — which is the collision the codes were split
            # for. Message to stderr, code 2: the run cannot produce a
            # record a reader can audit, and nothing was moved.
            print(f"FATAL: the gate record this run built violates the "
                  f"shared contract ({_ve}) — refusing to promote. The "
                  f"incumbent stands; the candidate is still in staging "
                  f"for post-mortem. Fix the writer; do not widen the "
                  f"reader.", file=sys.stderr)
            raise SystemExit(2) from None
        return _gate_arm, _gate

    def _live_incumbent() -> str:
        """The instruction production ACTUALLY runs for this signature.

        The gate must compare against what is deployed, not against the
        hard-coded seed: `result.baseline_instruction` is
        `signature.instruction`, which on any signature that has already been
        optimized is a DIFFERENT, unrelated string. Both sibling runners
        (optimize_verifier.py, optimize_tool_descriptions.py) already seed
        from and gate against the live artifact; this one did not.
        """
        try:
            data = _json_mod.loads(output_path.read_text(encoding="utf-8"))
            live = data.get("optimized_instruction")
            # `isinstance(str)` — matching optim/the loader's experiment-name note EXACTLY. A
            # bare `str(...)` accepted artifacts the loader rejects: an
            # artifact holding `42` became the 2-char baseline "42" here
            # while production ran the hand-written instruction, so any
            # candidate trivially "beat the live artifact" and shipped.
            # A gate that models a different production state than the one
            # that exists is worse than no gate.
            if isinstance(live, str) and live.strip():
                return live.strip()
        except Exception:
            pass
        return result.baseline_instruction

    # A/B ship-gate — ON BY DEFAULT. Only let the tuned prompt supersede the
    # baseline at inference if it actually wins on the held-out eval split.
    # Previously the gate was opt-in (`--ab-gate`) AND the tuned file was
    # written straight to the live path, so the documented invocation adopted
    # an UNPROVEN prompt globally (planning / tool-selection / reflection).
    # ⚠ BEFORE the early returns. Round 2 found this unreachable on the
    # `--no-ab-gate` path — the "did the oracle actually fire" guard was
    # skipped on exactly the path that adopts a prompt UNVERIFIED.
    _report_oracle_use()

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
        # ⚠ 2, NOT 1. Dead today (the pre-flight refuses an empty holdout
        # before the optimizer runs) and kept correct anyway — this
        # entry's history is guards that stopped being reachable when
        # something upstream moved, and a could-not-measure state
        # labelled "the gate rejected the candidate" is the exact
        # collision the codes were split for.
        return 2

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
        _fields = payload.get("expected_output") or {}
        want = _expected_target(_fields, sig)
        # Symmetric with the optimizer metric: the arms are compared on
        # the field the gold carries, not on everything the prompt
        # happened to emit.
        _gf = _gold_field(_fields, sig)
        if _gf:
            got = _section_of(got, _gf) or got
        passed = bool(want) and _overlap(want, got) >= _PASS_BAR
        return {"passed": passed, "output": got}

    incumbent = _live_incumbent()
    if incumbent != result.baseline_instruction:
        print(f"gating against the LIVE artifact ({len(incumbent)} chars), "
              f"not the seed baseline ({len(result.baseline_instruction)} chars)")

    # ⚠ NO CANDIDATE IS NOT A REJECTION — the sibling gate's round-16
    # lesson, ported. An optimizer that returns the incumbent verbatim
    # produced nothing: the two A/B arms would be byte-identical, every
    # pair concordant, delta exactly 0, and the run exited 1 — "the gate
    # rejected the candidate" — about a candidate that does not exist.
    # `GateExit.NO_CANDIDATE = 3` names this file in its docstring, and
    # until now this file had no site that could return it (lens C, C4i).
    # Checked BEFORE the A/B: two identical arms would also burn
    # 2 x len(private_set) main-model calls to measure a guaranteed zero.
    if (str(result.optimized_instruction or "").strip()
            == str(incumbent or "").strip()):
        _discard_staging()
        print("NO CANDIDATE: the optimizer returned the incumbent "
              "verbatim — there is nothing to promote and nothing to "
              "reject, and an A/B between two identical prompts would "
              "measure a guaranteed zero. This is a wasted run (or a "
              "broken reflection LM), not a verdict about the incumbent.",
              file=sys.stderr)
        return 3

    cmp = await compare_prompts(
        incumbent, result.optimized_instruction,
        private_set, _ab_runner, min_delta=args.ab_min_delta,
        # `compare_prompts` defaults to 30s and a timeout is scored as a
        # FAILED example, so the default made the verdict partly a latency
        # race — and it raced the two arms UNEQUALLY, because the arm that
        # produces more tokens is the slower one, which is exactly the arm a
        # prompt-length change moves. Measured 2026-08-04 re-running this
        # gate on the live 28-example planning.decompose private tier
        # (qwen-3.6-35b-a3 on the local upstream, no-think, max_tokens 8192):
        # 56 calls, median 1.2s incumbent / 4.0s candidate WARM, but the
        # cache-cold calls hit 12.3s / 32.2s and two warm calls reached
        # 27.5s and 28.5s — a 5% margin on the default, breached on the
        # cold head of every run. The ceiling only has to be above a real
        # stall: 8192 tokens at the measured ~25 tok/s is ~330s.
        per_example_timeout_s=360.0,
    )
    # ⚠ THE PRE-FLIGHT'S BAR MUST STILL HOLD AFTER THE RUN. `:564`
    # refuses to start below `_need` examples; round 5 made the DECIDING
    # tier the paired one, which can be far smaller, and nothing
    # re-checked it. Driven end to end: 50 examples cleared the
    # pre-flight, a 45-call outage left 5 usable pairs, delta +1.0000,
    # p=0.03125, SHIPS=True — promoted on 5 pairs. The sibling gate
    # refuses this exact shape (`ShipDecision.underpowered`).
    _n_paired = len(private_set) - cmp.transport_excluded
    # ⚠ A FLAG, NOT AN ASSIGNMENT. The first version set
    # `cmp.candidate_ships = False` here — and 45 lines below,
    # `_insignificant = (cmp.delta > min_delta and not
    # cmp.candidate_ships)` reads that same False as "the discordant
    # pairs were too few", so `--allow-insignificant-ship` set it back to
    # True. Driven end to end: 19 of 64 examples reached a verdict, the
    # guard printed "Nothing ships — --allow-insignificant-ship does NOT
    # override this", and the next line promoted. The message was
    # accurate about intent and false about behaviour.
    #
    # `_need` is the pre-flight's own bar, so `transport_excluded == 0`
    # already implies `_n_paired >= _need`; the condition is stated in
    # full anyway, because relying on that invariant is how the guard
    # silently disarms if the pre-flight ever moves.
    _below_evidence_bar = bool(cmp.transport_excluded
                               and _n_paired < _need)
    if _below_evidence_bar:
        print(f"⚠ EVIDENCE BELOW THE PRE-FLIGHT BAR: only {_n_paired} of "
              f"{len(private_set)} examples reached a verdict in BOTH "
              f"arms, under the {_need} this run was allowed to start on. "
              f"Nothing ships — re-run when the upstream is stable "
              f"(--allow-insignificant-ship does NOT override this; it "
              f"waives significance, not evidence).", file=sys.stderr)
        cmp.candidate_ships = False

    _p_str = ("n/a (no discordant pairs)" if cmp.p_value is None
              else f"{cmp.p_value:.4f}")
    # ⚠ n IS THE PAIRED COUNT, NOT THE TIER SIZE. §4DA round 5 made
    # `compare_prompts` decide over examples that reached a verdict in
    # BOTH arms — and left this line printing `len(private_set)` beside
    # the paired delta. Measured with a 10-call outage confined to one
    # arm: the delta is over 50 pairs and the line said n=60, with
    # `cmp.transport_excluded` printed nowhere. Round 5 diagnosed exactly
    # this shape ("each round leaves the instruments that report it one
    # revision behind") in the sibling gate and then did it here.
    _excl = ("" if not cmp.transport_excluded else
             f", {cmp.transport_excluded} of {len(private_set)} excluded "
             f"(no verdict in one or both arms; raw over all examples "
             f"{cmp.raw_baseline_pass_rate:.2f}/"
             f"{cmp.raw_candidate_pass_rate:.2f}, {cmp.raw_delta:+.2f})")
    print(f"A/B (PRIVATE holdout, n={_n_paired}{_excl}): "
          f"incumbent={cmp.baseline_pass_rate:.2f} "
          f"candidate={cmp.candidate_pass_rate:.2f} "
          f"delta={cmp.delta:+.2f} "
          f"McNemar p={_p_str} over "
          f"{cmp.baseline_wins + cmp.candidate_wins} discordant pairs "
          f"({cmp.candidate_wins} candidate / {cmp.baseline_wins} incumbent) "
          f"ships={cmp.candidate_ships}")

    # ⚠ THE MARGIN WAS THE WHOLE GATE, AND A MARGIN IS NOT A RESULT.
    # `candidate_ships` was `delta > min_delta` with no significance test.
    # The resolution guard above forces n >= ceil(1/min_delta), so at the
    # 0.02 default the smallest shipping swing was TWO examples out of 50
    # — which promotes 25-40% of the time under the null, depending on how
    # many pairs disagree. (An earlier version of this comment said ONE
    # example out of 31; that is unreachable, because the guard refuses
    # the run first. The conclusion held, the number did not.)
    # Meanwhile §4CW had already given the seed-arm VETO a significance
    # test. Shipping took two examples; refusing took a landslide.
    #
    # The bar now lives in `ab_eval.SHIP_ALPHA` and both directions read it.
    # ⚠ AND THE OVERRIDE MUST NOT SEE THE EVIDENCE GUARD'S False AS ITS
    # OWN. The sibling gate folds `not underpowered` into `cleared_margin`
    # and gates the override on that, which makes this structurally
    # unreachable; round 7 ported the sibling's MESSAGE and not its
    # STRUCTURE.
    _insignificant = (cmp.delta > args.ab_min_delta
                      and not cmp.candidate_ships
                      and not _below_evidence_bar)
    if _insignificant and args.allow_insignificant_ship:
        print("   --allow-insignificant-ship given; treating the margin as "
              "sufficient despite the discordant pairs.", file=sys.stderr)
        cmp.candidate_ships = True
        _ship_override[0] = True
    # ⚠ THE GATE RATCHETS, AND NOBODY WAS CHECKING WHERE IT RATCHETED TO.
    # `_live_incumbent()` makes each run "new candidate vs PREVIOUS
    # ARTIFACT", which is right for measuring an improvement and blind to
    # a slow drift away from the hand-written instruction the chain
    # started from. Nothing in three promotions ever asked "is any of
    # this better than the baseline?".
    #
    # Measured 2026-08-24 on `planning.decompose`: the 2026-07-29
    # artifact scored 0.071, the 2026-08-07 candidate 0.393 (+0.321 —
    # a real improvement, correctly promoted) — and the HAND-WRITTEN
    # BASELINE, never in either comparison, scores 0.484. Every
    # promotion was honest and the chain still walked away from the
    # thing it should have been beating.
    #
    # So the gate now runs a THIRD arm: the candidate must also not lose
    # to the seed instruction. Cheap (one more pass over the same private
    # tier) and it closes a hole that no amount of per-run rigour could.
    _seed = result.baseline_instruction
    # Only when the candidate would otherwise ship — a rejected candidate
    # does not need a second N-example pass to stay rejected.
    if (cmp.candidate_ships and _seed and _seed.strip()
            and _seed != incumbent):
        print(f"\nbaseline arm: candidate vs the HAND-WRITTEN seed "
              f"({len(_seed)} chars) — the arm the ratchet cannot see")
        _seed_cmp = await compare_prompts(
            _seed, result.optimized_instruction,
            private_set, _ab_runner, min_delta=0.0,
            # the same ceiling the main arm uses, for the same reason
            per_example_timeout_s=360.0)
        _seed_paired = len(private_set) - _seed_cmp.transport_excluded
        _seed_excl = ("" if not _seed_cmp.transport_excluded else
                      f"; {_seed_cmp.transport_excluded} of "
                      f"{len(private_set)} excluded (no verdict in one or "
                      f"both arms), raw over all examples "
                      f"{_seed_cmp.raw_baseline_pass_rate:.4f}/"
                      f"{_seed_cmp.raw_candidate_pass_rate:.4f}")
        print(f"  seed {_seed_cmp.baseline_pass_rate:.4f} vs candidate "
              f"{_seed_cmp.candidate_pass_rate:.4f} "
              f"(n={_seed_paired}{_seed_excl}; delta "
              f"{_seed_cmp.delta:+.4f}; candidate wins "
              f"{_seed_cmp.candidate_wins}, seed wins "
              f"{_seed_cmp.baseline_wins}, ties {_seed_cmp.ties})")

    # ⚠ THE REFUSAL NEEDS THE SAME NOISE FLOOR THE SHIP DIRECTION HAS.
    # Round 4 drove this end to end: written as `delta < 0`, a candidate
    # taking the artifact from 0.40 to 0.90 was THROWN AWAY over ONE
    # flipped example on a 31-tier — a 0.032 delta, smaller than the
    # gate's own 0.05 noise floor. A gate that calls a difference noise
    # in one direction and decisive in the other is calibrated on the
    # wrong statistic (§4BR), and this one would have welded the chain to
    # a terse baseline forever.
    # ⚠ THE VETO ARM NEEDS THE SAME OUTAGE HANDLING THE SHIP ARM GOT.
    # Rounds 5, 7 and 8 excluded unreached calls, reported the exclusion,
    # and refused below the pre-flight's bar — all on the MAIN arm, and
    # `_seed_cmp.transport_excluded` was read nowhere. It breaks BOTH
    # ways, driven on one corpus with only the seed arm's transport
    # changed:
    #   * total outage  -> "seed 0.0000 vs candidate 0.0000 (delta
    #     +0.0000 ... ties 0)", indistinguishable from a perfect tie, the
    #     VETO SUPPRESSED and the candidate PROMOTED over a seed it
    #     genuinely loses to (p=0.0000 when healthy);
    #   * partial outage leaving 5 of 45 -> delta -1.0000, p=0.0312, the
    #     veto MANUFACTURED and an honest promotion refused.
    # The exclusion is symmetric, so the direction is not manufactured by
    # the marker — the SAMPLE is. A veto is a refusal to ship, so an
    # underpowered one is not "safe": it welds the chain to whatever is
    # live, which is the failure §4CW added the seed arm to prevent.
    _seed_loses = False
    _seed_p = None
    _seed_underpowered = False
    if _seed_cmp is not None:
        _seed_underpowered = bool(
            _seed_cmp.transport_excluded
            and (len(private_set) - _seed_cmp.transport_excluded) < _need)
        if _seed_underpowered:
            print(f"  ⚠ SEED ARM BELOW THE PRE-FLIGHT BAR: only "
                  f"{len(private_set) - _seed_cmp.transport_excluded} of "
                  f"{len(private_set)} examples reached a verdict in both "
                  f"arms, under the {_need} this run started on. The seed "
                  f"veto is NOT DECIDABLE on this run — it is neither "
                  f"applied nor waived, and nothing ships, because a "
                  f"promotion whose seed arm was never measured is the "
                  f"ratchet §4CW exists to stop.", file=sys.stderr)
            cmp.candidate_ships = False
            _below_evidence_bar = True
        _b, _c = _seed_cmp.baseline_wins, _seed_cmp.candidate_wins
        # ONE implementation, in ab_eval — this used to be an inline copy,
        # and a second definition of the same statistic is how the two
        # halves of one gate drift apart (§4CW's `_PASS_BAR`, same shape).
        # One-sided in the veto's own direction — the question is "does
        # the SEED beat the candidate?", not "do they differ".
        _seed_p = ab_eval.mcnemar_p(_b, _c, alternative="baseline")
        # Refuse only when the seed beats the candidate by MORE than the
        # same margin the candidate needed to ship, AND the discordant
        # pairs support it. Either alone is a verdict without power.
        # ⚠ `not _seed_underpowered` IS REDUNDANT HERE AND KEPT ANYWAY.
        # The guard above already sets `cmp.candidate_ships = False`, and
        # the only reader of `_seed_loses` is gated on that — so dropping
        # this clause is an EQUIVALENT MUTANT today. It is kept because
        # the equivalence depends entirely on that ordering, and this
        # entry's own history is a list of guards that stopped being
        # reachable when something upstream moved. Stating the condition
        # where the verdict is computed costs nothing and survives the
        # reordering.
        _seed_loses = (not _seed_underpowered
                       and _seed_cmp.delta < -args.ab_min_delta
                       and _seed_p is not None
                       and _seed_p <= ab_eval.SHIP_ALPHA)
        print(f"  seed arm: candidate-minus-seed delta "
              f"{_seed_cmp.delta:+.4f} vs the "
              f"{-args.ab_min_delta:+.4f} refusal bar"
              + (f", McNemar p={_seed_p:.4f} over {_b + _c} discordant "
                 f"pairs" if _seed_p is not None else
                 " (no discordant pairs)"))

    if cmp.candidate_ships and _seed_loses:
        # ⚠ THE HEADLINE MUST MATCH WHAT HAPPENS. "⛔ NOT PROMOTING"
        # printed unconditionally, and `--allow-seed-loss` then promoted
        # four lines later — behaviour correct, headline false, which is
        # round 8's exact shape (a guard that REPORTS a refusal it does
        # not enforce). The verdict is decided first and the line says
        # which one it is.
        _seed_veto_stands = not args.allow_seed_loss
        print(f"\n{'⛔ NOT PROMOTING' if _seed_veto_stands else '⚠ SEED VETO OVERRIDDEN'}"
              f". The candidate beats the live artifact "
              f"({cmp.delta:+.4f}) but LOSES to the hand-written seed "
              f"({_seed_cmp.delta:+.4f}, McNemar p={_seed_p:.4f}). "
              f"Promoting it would ratchet the chain further from the "
              f"instruction it should be beating — which is how the live "
              f"artifact came to be serving every planner turn with no "
              f"valid measured win.\n"
              f"   If the seed is genuinely worse for this signature, say "
              f"so with a measurement and re-run with --allow-seed-loss.",
              file=sys.stderr)
        if _seed_veto_stands:
            _discard_staging()
            return 1
        print("   --allow-seed-loss given; promoting anyway.",
              file=sys.stderr)
        # Recorded IN the artifact, not only on stderr: an override that
        # leaves no trace in the thing it overrode is an override nobody
        # can audit later.
        _seed_override[0] = True

    if not cmp.candidate_ships:
        _discard_staging()
        if _below_evidence_bar:
            # ⚠ NAME THE CAUSE THAT ACTUALLY FIRED, AND THE ARM IT FIRED
            # ON. The insignificance branch below printed "McNemar
            # p=0.0000 > 0.05" — an arithmetic falsehood; the pairs DID
            # support it — and then told the operator to override with a
            # flag that (correctly) cannot help. And a seed-arm outage
            # reported the MAIN arm's healthy counts, which name the
            # wrong arm to go and fix.
            if _seed_underpowered:
                _arm_n = (len(private_set)
                          - _seed_cmp.transport_excluded)
                print(f"A/B gate ABORTED: the SEED ARM "
                      f"(candidate vs the hand-written seed) reached a "
                      f"verdict on only {_arm_n} of {len(private_set)} "
                      f"examples, under the {_need} the pre-flight "
                      f"required. The main arm was fine "
                      f"({cmp.delta:+.4f} over {_n_paired} pairs) — it is "
                      f"the VETO that could not be decided, and shipping "
                      f"without it is the ratchet the seed arm exists to "
                      f"stop. Re-run when the upstream is stable.")
            else:
                print(f"A/B gate ABORTED: only {_n_paired} "
                      f"of {len(private_set)} examples reached a verdict "
                      f"in both arms, under the {_need} the pre-flight "
                      f"required. This is a TRANSPORT failure, not a "
                      f"measured loss — the margin ({cmp.delta:+.4f}) and "
                      f"the pairs ({cmp.candidate_wins} candidate / "
                      f"{cmp.baseline_wins} incumbent) are computed over "
                      f"too little evidence to act on. Re-run when the "
                      f"upstream is stable.")
        elif _insignificant:
            # Distinguishing the two rejections matters: "clear the margin"
            # and "collect more evidence" are different instructions, and a
            # single message would send the reader to re-tune a prompt that
            # may already be better.
            # No `>` glyph: `delta` is a float difference of ratios, so
            # 2/100 - 0/100 is 0.020000000000000018 and ANY rounding that
            # matches the bar's own precision prints a comparison that
            # reads false. State the two numbers; let the reader compare.
            print(f"A/B gate REJECTED the candidate: it cleared the margin "
                  f"(delta {cmp.delta:+.4f}, bar {args.ab_min_delta}) but the "
                  f"discordant pairs do not support it "
                  f"(McNemar p={_p_str} > {ab_eval.SHIP_ALPHA}, "
                  f"{cmp.candidate_wins} candidate / {cmp.baseline_wins} "
                  f"incumbent). This is an UNDERPOWERED verdict, not a "
                  f"measured loss — the holdout is n={_n_paired}. "
                  f"Collect more graded turns, or override deliberately "
                  f"with --allow-insignificant-ship.")
        else:
            print(f"A/B gate REJECTED the candidate (delta "
                  f"{cmp.delta:+.4f}, bar {args.ab_min_delta}); discarded "
                  f"staging — baseline stands.")
        # ⚠ AN ABORT IS NOT A REJECTION. The evidence-bar and seed-arm
        # outage branches print "re-run when the upstream is stable" —
        # nothing was measured there, so exiting 1 ("the gate rejected
        # the candidate") was the same collision the judges' codes were
        # split for. Lens C, C4(iii): both gates shared this.
        return 2 if _below_evidence_bar else 1
    _promote_staging()
    print(f"A/B gate PASSED — candidate promoted to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
