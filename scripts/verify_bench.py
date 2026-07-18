#!/usr/bin/env python3
"""Verifier fault-injection calibration bench — CLI runner.

Measures the verifier's actual catch rate (TPR) and false-alarm rate
(FPR) per corruption class, and A/Bs the two-stage forced-identification
prompt (GHOST_VERIFY_TWO_STAGE) against the classic single-prompt path.
See src/ghost_agent/eval/verify_bench.py for the methodology.

The judge endpoint is the system under test — point --base-url at the
model that actually serves VERIFY in production (the worker node's
llama-server, e.g. nova, or the main box's):

    PYTHONPATH=src python scripts/verify_bench.py \
        --base-url http://nova:8081 --two-stage both

Cases come from the checked-in seed set plus (optionally) real turns
minted from GHOST_LLM_RECORD day-files:

    python scripts/verify_bench.py --base-url http://127.0.0.1:8080 \
        --recordings "$GHOST_HOME/system/llm_recordings" --max-cases 40

Outputs verify_bench_out/<UTC-ts>/results.json + report.md and prints
the report. Nothing here touches the live agent or its data.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ghost_agent.core.verifier import Verifier  # noqa: E402
from ghost_agent.eval.verify_bench import (  # noqa: E402
    FAULTS,
    HttpChatClient,
    extract_cases_from_recordings,
    load_cases_jsonl,
    render_report_md,
    run_bench,
)

DEFAULT_CASES = REPO_ROOT / "scripts" / "verify_bench_cases.jsonl"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080",
                    help="OpenAI-compatible endpoint of the judge model "
                         "(the model that serves VERIFY in production)")
    ap.add_argument("--model", default="",
                    help="model name to send, if the endpoint needs one")
    ap.add_argument("--api-key", default="",
                    help="bearer token, if the endpoint needs one")
    ap.add_argument("--cases", default=str(DEFAULT_CASES),
                    help="seed cases JSONL (claim/evidence/context)")
    ap.add_argument("--recordings", default="",
                    help="dir or file of GHOST_LLM_RECORD day-files to "
                         "mint extra cases from real turns")
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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="parallel verify calls; keep 1 for a "
                         "single-slot llama-server")
    ap.add_argument("--timeout", type=float, default=90.0,
                    help="per-request timeout seconds")
    ap.add_argument("--out", default="verify_bench_out",
                    help="output root directory")
    return ap.parse_args()


async def _amain() -> int:
    args = _parse_args()

    cases = load_cases_jsonl(args.cases)
    if args.recordings:
        rec = Path(args.recordings)
        paths = sorted(rec.glob("*.jsonl")) if rec.is_dir() else [rec]
        minted = extract_cases_from_recordings(paths)
        print(f"minted {len(minted)} case(s) from recordings")
        cases.extend(minted)
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

    client = HttpChatClient(args.base_url, timeout=args.timeout,
                            api_key=args.api_key, model=args.model)
    verifier = Verifier(llm_client=client)

    done = {"n": 0}

    def _progress(res) -> None:
        done["n"] += 1
        v = res.verdict or ("ERROR" if res.error else "SKIP")
        print(f"  [{done['n']:>3}] {res.trial.case_id:<22} "
              f"{res.trial.fault:<22} -> {v:<9} "
              f"conf={res.confidence:.2f} {res.elapsed_s:5.1f}s"
              + (f"  ({res.error[:60]})" if res.error else ""))

    print(f"{len(cases)} case(s), arms: {', '.join(arms)}, "
          f"judge: {args.base_url}")
    try:
        report = await run_bench(
            cases, verifier, arms=arms, fault_names=fault_names,
            seed=args.seed, concurrency=args.concurrency,
            on_result=_progress)
    finally:
        await client.aclose()

    report["judge_base_url"] = args.base_url
    report["judge_model"] = args.model
    report["cases_file"] = str(args.cases)
    report["recordings"] = str(args.recordings)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_report_md(report)
    (out_dir / "report.md").write_text(md, encoding="utf-8")

    print()
    print(md)
    print(f"written: {out_dir}/results.json  {out_dir}/report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
