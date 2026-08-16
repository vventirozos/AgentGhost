#!/usr/bin/env python3
"""Verifier fault-injection calibration bench — CLI runner.

Measures the verifier's actual catch rate (TPR) and false-alarm rate
(FPR) per corruption class, and A/Bs the two-stage forced-identification
prompt (GHOST_VERIFY_TWO_STAGE) against the classic single-prompt path.
See src/ghost_agent/eval/verify_bench.py for the methodology.

The judge endpoint is the system under test — point --base-url at the
model that actually serves VERIFY in production. Measured on the live
process 2026-08-04 (`ps eww`): the VERIFY worker is
`--worker-nodes http://100.83.184.117:8088|Nova` (Gemma 4 E4B) and the
main model is `--upstream-url http://127.0.0.1:8088` (Qwen3.6-35B);
`--critic-nodes` was not set THEN, so verify rode the WORKER route.
⚠ Since 2026-08-06 the live process boots `--critic-nodes`, so today's
cheap judge rides the CRITIC branch (120s ceiling, one extra fallback
rung) — pass `--leg critic` (the default) to mirror that, or
`--leg worker` to reproduce the 2026-08-04 topology. The leg is
recorded in `bench_provenance.escalation.route_health.leg`; check it
before comparing two reports.

TWO ARMS — say which one you want, because they measure different
systems:

  raw judge (cheap judge standalone; `_escalate_refute` cannot fire):
    PYTHONPATH=src python scripts/verify_bench.py \
        --base-url http://100.83.184.117:8088 --two-stage both

  judge+escalation (what production acts on — cheap judge screens, main
  model re-adjudicates every REFUTED):
    PYTHONPATH=src python scripts/verify_bench.py \
        --base-url http://100.83.184.117:8088 \
        --main-base-url http://127.0.0.1:8088 --two-stage both

Only the escalated arm produces a production-comparable false-alarm
rate. On the live recorded corpus (2026-07-30..08-04) the main model
overturned 42 of 50 (84%) of the cheap judge's refutes, so the raw arm's
number is an upper bound; the report names it `fpr_raw_judge` and refuses
to call it FPR.

Cases come from the checked-in seed set plus (optionally) real turns
minted from GHOST_LLM_RECORD day-files:

    python scripts/verify_bench.py --base-url http://100.83.184.117:8088 \
        --recordings "$GHOST_HOME/system/llm_recordings" --max-cases 40

Outputs verify_bench_out/<UTC-ts>/results.json + report.md and prints
the report. Nothing here touches the live agent or its data.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ghost_agent.core.verifier import Verifier  # noqa: E402
from ghost_agent.eval.runprogress import RunProgress  # noqa: E402
from ghost_agent.eval.verify_bench import (  # noqa: E402
    ARM_ESCALATED,
    FAULTS,
    EscalatingChatClient,
    HttpChatClient,
    ResponseCache,
    build_case_pool,
    escalation_arm,
    extract_cases_from_recordings,
    load_cases_jsonl,
    render_report_md,
    run_bench,
)

DEFAULT_CASES = REPO_ROOT / "scripts" / "verify_bench_cases.jsonl"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8088",
                    help="OpenAI-compatible endpoint of the judge model "
                         "(the model that serves VERIFY in production — "
                         "live: the Nova worker). The old 127.0.0.1:8080 "
                         "default answered nothing at all.")
    ap.add_argument("--model", default="",
                    help="model name to send, if the endpoint needs one")
    ap.add_argument("--api-key", default="",
                    help="bearer token, if the endpoint needs one")
    ap.add_argument("--main-base-url", default="",
                    help="endpoint of the MAIN model the refute-escalation "
                         "adjudicates on (live: http://127.0.0.1:8088). "
                         "Setting it runs the `judge+escalation` arm — the "
                         "pipeline production acts on, and the ONLY arm "
                         "whose false-alarm rate is a production FPR. "
                         "Unset = raw cheap judge, reported as "
                         "`fpr_raw_judge` and never as FPR.")
    ap.add_argument("--main-model", default="",
                    help="model name for --main-base-url, if it needs one")
    ap.add_argument("--cases", default=str(DEFAULT_CASES),
                    help="seed cases JSONL (claim/evidence/context)")
    ap.add_argument("--mined-cases", default="",
                    help="recording-derived case pool (default "
                         "$GHOST_HOME/system/eval/verify_bench_cases_mined.jsonl "
                         "when present). Kept out of the repo: derived from "
                         "live turns.")
    ap.add_argument("--no-mined", action="store_true",
                    help="seed cases only — reproduces the pre-2026-08-04 "
                         "pool, whose private tier cannot resolve a 0.02 "
                         "ship gate")
    ap.add_argument("--recordings", default="",
                    help="dir or file of GHOST_LLM_RECORD day-files to "
                         "mint extra cases from real turns")
    ap.add_argument("--tier", choices=("all", "public", "private"),
                    default="all",
                    help="restrict mined cases to a holdout tier. Use "
                         "'private' to bench AFTER an optimize_verifier run: "
                         "that script trains on the PUBLIC tier of this same "
                         "pool, so the default 'all' measures partly on cases "
                         "the optimizer saw.")
    ap.add_argument("--force", action="store_true",
                    help="with --refresh-mined, allow a fresh mint that "
                         "shrinks the pool by more than half")
    ap.add_argument("--refresh-mined", action="store_true",
                    help="re-mint the mined pool from --recordings (default "
                         "$GHOST_HOME/system/llm_recordings), WRITE it to the "
                         "mined-cases path, and exit. The pool is a durable "
                         "artifact that no script could previously rebuild, so "
                         "it silently kept whatever redaction/extraction bugs "
                         "were live the day it was minted.")
    ap.add_argument("--skip-cases", type=int, default=0,
                    help="skip the first N cases (split a slow arm "
                         "across multiple bounded runs)")
    ap.add_argument("--max-cases", type=int, default=0,
                    help="cap total cases (0 = no cap)")
    ap.add_argument("--faults", default="",
                    help=f"comma-separated subset of: {', '.join(FAULTS)}")
    ap.add_argument("--two-stage", choices=("on", "off", "both"),
                    default="both",
                    help="which prompt arm(s) to run (default both)")
    ap.add_argument("--logit-expect", choices=("on", "off", "inherit"),
                    default="inherit",
                    help="pin GHOST_VERIFY_LOGIT_EXPECT for the run "
                         "(§4BF flip i); 'inherit' leaves the shell "
                         "value in force like every other discipline "
                         "flag. Probe calls are cheap-pool; they miss "
                         "the replay cache on a FIRST probe-armed run "
                         "(new payload shape) and replay from it on "
                         "re-runs — a crashed-and-restarted arm replays "
                         "its own recorded probe responses.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel verify calls; keep 1 for a "
                         "single-slot llama-server")
    ap.add_argument("--leg", choices=("critic", "worker"),
                    default="critic",
                    help="which production rung serves the cheap judge "
                         "(critic = live topology since 2026-08-06)")
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="per-request timeout seconds")
    ap.add_argument("--out", default="verify_bench_out",
                    help="output root directory")
    # ── Re-bench economics. A full live run is hundreds of judge calls
    # plus a main-model adjudication per REFUTED; paying that on every
    # code change is why the number goes stale and gets quoted anyway.
    ap.add_argument("--cache-mode", choices=("off", "write", "read", "strict"),
                    default="off",
                    help="off: fresh live run (the honest 'number today'). "
                         "write: live, but record every response. "
                         "read: replay hits, live-call misses (INCREMENTAL — "
                         "after a prompt edit only changed prompts cost). "
                         "strict: replay only, a miss is an error (PROVES "
                         "the delta is attributable to code, not the model)")
    # ⚠ ANCHORED TO $GHOST_HOME, not to the CWD (2026-08-10).
    #
    # This was the relative string "verify_bench_cache". Run from the repo
    # root — the normal way — it resolved to <repo>/verify_bench_cache, a
    # directory that did not exist, which read/strict then silently CREATED
    # and missed on forever, while the real 2389-entry cache sat untouched
    # in $GHOST_HOME/system/eval. "0 hits" from an empty cache reads exactly
    # like "0 hits" from an invalidated one. Same anchoring as the mined
    # case pool below; the relative name survives only as a fallback for
    # environments with no GHOST_HOME.
    _cache_default = (
        str(Path(os.environ["GHOST_HOME"]) / "system" / "eval"
            / "verify_bench_cache")
        if os.getenv("GHOST_HOME") else "verify_bench_cache")
    ap.add_argument("--cache-dir", default=_cache_default,
                    help="where cached judge responses live (default: "
                         "$GHOST_HOME/system/eval/verify_bench_cache)")
    ap.add_argument("--progress-file", default="",
                    help="where to write live progress JSON (default: "
                         "<out>/progress.json). Read it with "
                         "scripts/runstatus.py — never infer progress from "
                         "cache counts or log parsing.")
    return ap.parse_args()


async def _amain() -> int:
    args = _parse_args()

    cases = load_cases_jsonl(args.cases)
    # Mined pool, kept OUTSIDE the repo: it derives from live user turns, so
    # even redacted it is operator data rather than a checked-in fixture.
    # This is where the volume is — the 21 seed cases give a private tier
    # too coarse to resolve the ship gate's own --min-delta.
    _mined_default = (Path(os.getenv("GHOST_HOME", "")) / "system" / "eval"
                      / "verify_bench_cases_mined.jsonl"
                      ) if os.getenv("GHOST_HOME") else None
    _mined_path = Path(args.mined_cases) if args.mined_cases else _mined_default

    if args.refresh_mined:
        if not _mined_path:
            print("--refresh-mined needs $GHOST_HOME or --mined-cases",
                  file=sys.stderr)
            return 2
        _rec = Path(args.recordings) if args.recordings else (
            Path(os.getenv("GHOST_HOME", "")) / "system" / "llm_recordings")
        paths = sorted(_rec.glob("*.jsonl")) if _rec.is_dir() else [_rec]
        if not paths:
            print(f"no recordings under {_rec}", file=sys.stderr)
            return 2
        for _flag, _val in (("--max-cases", args.max_cases),
                            ("--skip-cases", args.skip_cases),
                            ("--no-mined", args.no_mined)):
            if _val:
                print(f"{_flag} does not apply to --refresh-mined (it mints "
                      f"the WHOLE pool). Remove it.", file=sys.stderr)
                return 2
        minted = extract_cases_from_recordings(paths)

        # ── FLOOR CHECK, before anything is written ────────────────────
        # Silent extraction failure is THE failure mode of this pipeline —
        # a retuned verifier template once matched 0 of 580 records while the
        # opening sentence matched perfectly. Overwriting a durable pool with
        # the result of that is how a bad day becomes a lost artifact: an
        # empty pool takes optimize_verifier's private tier back to 4 cases /
        # step 0.0833, i.e. straight back to REFUSING TO RUN.
        _old = []
        if _mined_path.exists():
            try:
                _old = [l for l in _mined_path.read_text(
                    encoding="utf-8").splitlines() if l.strip()]
            except OSError:
                pass
        if not minted:
            print(f"REFUSING TO WRITE: minted 0 cases from {len(paths)} "
                  f"recording file(s). That is an extraction failure, not an "
                  f"empty corpus — check that the live verify templates still "
                  f"invert. Existing pool ({len(_old)} rows) left untouched.",
                  file=sys.stderr)
            return 2
        if _old and len(minted) < 0.5 * len(_old) and not args.force:
            print(f"REFUSING TO WRITE: the fresh mint has {len(minted)} cases "
                  f"against {len(_old)} already in the pool — a >50% drop is "
                  f"far more likely to be a broken extractor than real corpus "
                  f"churn. Re-run with --force if the shrink is intended.",
                  file=sys.stderr)
            return 2

        _mined_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = _mined_path.with_name(_mined_path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for c in minted:
                # `high_stakes` is deliberately NOT persisted. Mined cases
                # stay in DERIVE mode (the field absent => None), so the
                # flag is recomputed per trial from that trial's evidence.
                # Freezing the derived value here would pin every trial to
                # the clean case's stakes and stop `silent_failure` — the
                # fault that turns evidence into a tool error, i.e. exactly
                # `_escalate_confirm`'s population — from ever reaching the
                # CONFIRM direction. Only a hand-authored case should pin it.
                fh.write(json.dumps({
                    "case_id": c.case_id, "claim": c.claim,
                    "evidence": c.evidence, "context": c.context,
                    "source": c.source, "notes": c.notes,
                }, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(_mined_path)
        ids = {c.case_id for c in minted}
        print(f"re-minted {len(minted)} case(s) ({len(ids)} distinct case_id) "
              f"-> {_mined_path}  [was {len(_old)} rows]")
        if len(ids) != len(minted):
            print("WARNING: duplicate case_id in the fresh pool — "
                  "bench_provenance becomes order-dependent", file=sys.stderr)
        return 0

    if _mined_path and _mined_path.exists() and not args.no_mined:
        # ONE implementation, shared with scripts/verify_bench_status.py.
        # This block used to be the only copy; the oracle carried a replica
        # that computed 35 cases where this loaded 58 (it tier-filtered the
        # SEED set, which this does not, and skipped the mined-vs-seed
        # dedup), so the fingerprint reported permanent false drift.
        from ghost_agent.optim.trainset import holdout_tier
        _before = len(cases)
        cases = build_case_pool(args.cases, _mined_path, False, args.tier)
        extra = cases[_before:]
        n_pub = sum(1 for c in extra
                    if holdout_tier(f"vbcase:{c.case_id}",
                                    private_pct=30) == "public")
        print(f"loaded {len(extra)} mined case(s) from {_mined_path} "
              f"(tier={args.tier})")
        if args.tier == "all" and n_pub:
            print(f"  NOTE: {n_pub} of them are in the PUBLIC tier that "
                  f"optimize_verifier.py trains on. For a clean post-"
                  f"optimization measurement use --tier private.")
    if args.recordings:
        rec = Path(args.recordings)
        paths = sorted(rec.glob("*.jsonl")) if rec.is_dir() else [rec]
        minted = extract_cases_from_recordings(paths)
        print(f"minted {len(minted)} case(s) from recordings")
        have = {(c.claim.strip(), c.evidence.strip()) for c in cases}
        cases.extend([c for c in minted
                      if (c.claim.strip(), c.evidence.strip()) not in have])
    if args.skip_cases > 0:
        cases = cases[args.skip_cases:]
    if args.max_cases > 0:
        cases = cases[:args.max_cases]
    if not cases:
        print("no cases to run", file=sys.stderr)
        return 2

    fault_names = ([f.strip() for f in args.faults.split(",") if f.strip()]
                   or None)
    arms = {"on": ["two_stage_on"], "off": ["two_stage_off"],
            "both": ["two_stage_on", "two_stage_off"]}[args.two_stage]

    # ARM. With --main-base-url the client exposes the same two legs the
    # production LLMClient does (worker route + main chat_completion), so
    # `Verifier._escalate_refute` fires exactly as it does live. Without
    # it, escalation is structurally impossible and the report says so.
    cache = ResponseCache(args.cache_dir or None, args.cache_mode)
    if args.cache_mode != "off":
        # ⚠ RESOLVED ABSOLUTE PATH + ENTRY COUNT, not the basename. The old
        # line printed "dir=verify_bench_cache", which is equally true of the
        # right cache and of an empty one accidentally created in the CWD —
        # the bug was on screen the whole time and unreadable.
        _n = cache.entry_count()
        _where = cache.path.resolve() if cache.path else "(none)"
        print(f"  response cache: mode={args.cache_mode} dir={_where} "
              f"({_n} entries)")
        if args.cache_mode in ("read", "strict"):
            if _n == 0:
                print("  ⚠⚠ CACHE IS EMPTY — every lookup will miss, and a "
                      "replay against an empty cache measures NOTHING. "
                      "Check --cache-dir.")
            print("  ⚠ a replayed report measures CODE against a FROZEN "
                  "judge — it is not 'the number today'")
    if args.main_base_url:
        client = EscalatingChatClient(
            args.base_url, args.main_base_url, timeout=args.timeout,
            api_key=args.api_key, model=args.model,
            main_model=args.main_model, leg=args.leg, cache=cache)
    else:
        client = HttpChatClient(args.base_url, timeout=args.timeout,
                                api_key=args.api_key, model=args.model,
                                cache=cache)
    verifier = Verifier(llm_client=client)

    done = {"n": 0}

    # PROGRESS CONTRACT (2026-08-09). This counter used to exist only as a
    # print() to a block-buffered stdout, so a redirected run was blind for
    # its whole duration and progress had to be GUESSED from side effects —
    # cache file counts, call-shape ratios, log regexes. Every one of those
    # guesses was wrong. The position is now written to a file that any
    # reader can consult, and `--progress-file` defaults next to --out so
    # observability is the default rather than something to remember.
    # ⚠ Constructed AFTER `_trials` exists (below), not here: the callback is
    # defined before the trials are built, so the TOTAL is not knowable yet.
    # A first version read `len(trials)` at this point and died with
    # NameError on every run — caught in 0.2s by a strict replay rather than
    # 90 minutes into a live one, which is the whole argument for replay.
    _prog_holder = {}

    def _progress(res) -> None:
        done["n"] += 1
        v = res.verdict or ("ERROR" if res.error else "SKIP")
        print(f"  [{done['n']:>3}] {res.trial.case_id:<22} "
              f"{res.trial.fault:<22} -> {v:<9} "
              f"conf={res.confidence:.2f} {res.elapsed_s:5.1f}s"
              + (f"  ({res.error[:60]})" if res.error else ""), flush=True)
        _p = _prog_holder.get("p")
        if _p is not None:
            _p.tick(extra={"last_case": res.trial.case_id,
                           "last_fault": res.trial.fault,
                           "last_verdict": v})

    # Built here (not inside run_bench) only so the banner can report the
    # high-stakes count before the run starts; run_bench rebuilds them
    # from the same (cases, faults, seed), which is a pure function.
    from ghost_agent.eval.verify_bench import build_trials
    _trials = build_trials(cases, fault_names=fault_names, seed=args.seed)
    _esc = escalation_arm(verifier, _trials)
    # The DENOMINATOR comes from the built trials, never from cases*faults —
    # not every fault injects into every case (fact_swap applied to 43 of 58
    # on 2026-08-09), so 58*8 overstates the total by 31 and every derived
    # percentage with it.
    _prog_holder["p"] = RunProgress(
        args.progress_file or (Path(args.out) / "progress.json"),
        total=len(_trials) * len(arms),
        label=f"verify_bench {args.tier}/{args.two_stage}")
    print(f"{len(cases)} case(s), arms: {', '.join(arms)}, "
          f"judge: {args.base_url}")
    # Echo the probe pin (R1 review: a mislaunched arm — 'inherit'
    # where 'on' was intended — was only discoverable post-hoc in
    # provenance; the banner is where the operator actually looks).
    print(f"logit-expect probe: {args.logit_expect}"
          + ("" if args.logit_expect != "inherit" else
             f" (shell: GHOST_VERIFY_LOGIT_EXPECT="
             f"{os.environ.get('GHOST_VERIFY_LOGIT_EXPECT', '<unset>')})"))
    print(f"verdict pipeline: {_esc['arm']}"
          + (f" (escalates to {args.main_base_url})"
             if _esc.get("main") else ""))
    for _name, _d in (_esc.get("directions") or {}).items():
        print(f"  escalate-{_name:<8} {'LIVE' if _d['live'] else 'dark'}"
              + (f" — {_d['why_not']}" if not _d["live"] else "")
              + (f" ({_d['high_stakes_trials']} high-stakes trials)"
                 if _name == "confirm"
                 and _d.get("high_stakes_trials") is not None else ""))
    if _esc["arm"] != ARM_ESCALATED:
        print(f"  NOTE: this run measures {_esc['measures']}. Pass "
              f"--main-base-url http://127.0.0.1:8088 (and leave both "
              f"GHOST_VERIFY_ESCALATE_* switches on) for the "
              f"production-equivalent arm.")
    if _esc.get("confirm_unexercised"):
        print(f"  NOTE: {_esc['confirm_unexercised']}.")
    try:
        report = await run_bench(
            cases, verifier, arms=arms, fault_names=fault_names,
            seed=args.seed, concurrency=args.concurrency,
            on_result=_progress,
            logit_expect=(None if args.logit_expect == "inherit"
                          else args.logit_expect))
    finally:
        await client.aclose()

    report["judge_base_url"] = args.base_url
    report["judge_model"] = args.model
    report["cases_file"] = str(args.cases)
    report["recordings"] = str(args.recordings)
    # The authoritative arm record lives in provenance.escalation (written
    # by run_bench from the CLIENT, not from these flags). These two are
    # the flags as typed — kept so a mismatch between "what was asked for"
    # and "what actually ran" is visible instead of reconciled.
    report["main_base_url"] = args.main_base_url
    report["main_model"] = args.main_model

    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_report_md(report)
    (out_dir / "report.md").write_text(md, encoding="utf-8")

    # ⚠ WITHOUT THIS every completed run read as STALLED forever: nothing
    # ever recorded a terminal state, so `read_progress` aged the last tick
    # into "the run may be wedged". Found by fresh-eyes audit, not by a test.
    _p = _prog_holder.get("p")
    if _p is not None:
        _p.finish(f"results in {out_dir}")

    print()
    print(md)
    print(f"written: {out_dir}/results.json  {out_dir}/report.md")
    cs = report.get("provenance", {}).get("cache", {})
    if cs.get("mode") != "off":
        print(f"cache: {cs.get('hits')} hits / {cs.get('misses')} misses / "
              f"{cs.get('writes')} writes — {cs.get('measures')}")
    # LOUD and last, because a strict miss degrades silently into a null
    # verdict: the scores print normally and are quietly computed over
    # fewer trials. A replay that did not replay must not look clean.
    if cache.strict_misses:
        dumped = cache.dump_misses(out_dir / "strict_cache_misses.json")
        print(f"\n⚠⚠ {len(cache.strict_misses)} STRICT CACHE MISS(ES) — this "
              f"run is NOT a clean replay.\n   The missing requests are in "
              f"{dumped}\n   Re-run with --cache-mode read to fill them with "
              f"live calls, then replay again.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
