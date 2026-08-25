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
import json as _json_mod
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
    _resolution = 1.0 / max(1, len(private_set))
    if _resolution > args.ab_min_delta:
        _need = math.ceil(1.0 / args.ab_min_delta)
        print(f"REFUSING TO RUN: the A/B gate cannot resolve its own "
              f"threshold. {len(private_set)} private examples give a "
              f"smallest step of {_resolution:.3f}, coarser than "
              f"--ab-min-delta {args.ab_min_delta}. One flipped example "
              f"would decide the run. Collect at least {_need} private "
              f"examples, or raise --ab-min-delta.", file=sys.stderr)
        return 1

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

    def _promote_staging():
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
            # `cmp` exists only on the GATED path (R2 review: the
            # --no-ab-gate promotion hit a NameError here, was swallowed,
            # and every ungated promotion shipped with the "promoted
            # without provenance" warning instead of an honest stamp).
            try:
                _cmp = cmp
            except NameError:
                _cmp = None
            if _cmp is not None:
                _art["gate_arm"] = "token-F1 A/B, private holdout"
                _art["gate"] = {
                    "metric": "token_f1_overlap>=0.3",
                    "seed_arm": (None if _seed_cmp is None else {
                        "seed_pass_rate": _seed_cmp.baseline_pass_rate,
                        "candidate_pass_rate": _seed_cmp.candidate_pass_rate,
                        "delta": _seed_cmp.delta,
                        "seed_wins": _seed_cmp.baseline_wins,
                        "candidate_wins": _seed_cmp.candidate_wins,
                        "overridden": _seed_override[0],
                    }),
                    "n_private": len(private_set),
                    "incumbent_pass_rate": round(_cmp.baseline_pass_rate, 4),
                    "candidate_pass_rate": round(_cmp.candidate_pass_rate, 4),
                    "delta": round(_cmp.delta, 4),
                    "min_delta": args.ab_min_delta,
                    "promoted_utc": __import__("time").strftime(
                        "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                }
            else:
                _art["gate_arm"] = "UNGATED (--no-ab-gate)"
                _art["gate"] = {
                    "metric": "none — adopted unverified",
                    "promoted_utc": __import__("time").strftime(
                        "%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
                }
            output_path.write_text(_json_mod.dumps(_art, indent=1),
                                   encoding="utf-8")
        except Exception as _se:  # noqa: BLE001 — stamp must not unship
            print(f"WARNING: gate stamp failed ({_se}) — artifact "
                  f"promoted without provenance", file=sys.stderr)


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
            # `isinstance(str)` — matching optim/loader.py:69 EXACTLY. A
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
    print(f"A/B (PRIVATE holdout, n={len(private_set)}): "
          f"incumbent={cmp.baseline_pass_rate:.2f} "
          f"candidate={cmp.candidate_pass_rate:.2f} "
          f"delta={cmp.delta:+.2f} ships={cmp.candidate_ships}")
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
        print(f"  seed {_seed_cmp.baseline_pass_rate:.4f} vs candidate "
              f"{_seed_cmp.candidate_pass_rate:.4f} "
              f"(delta {_seed_cmp.delta:+.4f}; candidate wins "
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
    _seed_loses = False
    _seed_p = None
    if _seed_cmp is not None:
        _b, _c = _seed_cmp.baseline_wins, _seed_cmp.candidate_wins
        if _b + _c:
            from math import comb as _comb
            _nd, _k = _b + _c, min(_b, _c)
            _seed_p = min(1.0, sum(_comb(_nd, i)
                                   for i in range(_k + 1)) / (2 ** _nd) * 2)
        # Refuse only when the seed beats the candidate by MORE than the
        # same margin the candidate needed to ship, AND the discordant
        # pairs support it. Either alone is a verdict without power.
        _seed_loses = (_seed_cmp.delta < -args.ab_min_delta
                       and _seed_p is not None and _seed_p <= 0.05)
        print(f"  seed arm: delta {_seed_cmp.delta:+.4f} vs the "
              f"{-args.ab_min_delta:+.4f} refusal bar"
              + (f", McNemar p={_seed_p:.4f} over {_b + _c} discordant "
                 f"pairs" if _seed_p is not None else
                 " (no discordant pairs)"))

    if cmp.candidate_ships and _seed_loses:
        print(f"\n⛔ NOT PROMOTING. The candidate beats the live artifact "
              f"({cmp.delta:+.4f}) but LOSES to the hand-written seed "
              f"({_seed_cmp.delta:+.4f}, McNemar p={_seed_p:.4f}). "
              f"Promoting it would ratchet the chain further from the "
              f"instruction it should be beating — which is how the live "
              f"artifact came to be serving every planner turn with no "
              f"valid measured win.\n"
              f"   If the seed is genuinely worse for this signature, say "
              f"so with a measurement and re-run with --allow-seed-loss.",
              file=sys.stderr)
        if not args.allow_seed_loss:
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
        print(f"A/B gate REJECTED the candidate (delta {cmp.delta:+.2f} "
              f"≤ {args.ab_min_delta}); discarded staging — baseline stands.")
        return 1
    _promote_staging()
    print(f"A/B gate PASSED — candidate promoted to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
