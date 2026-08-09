#!/usr/bin/env python3
"""Is the recorded verifier bench number still valid? — the staleness oracle.

THE PROBLEM. A full live bench is hundreds of judge calls plus a main-model
adjudication on every REFUTED. Nobody pays that on each code change, so in
practice the last number gets quoted long after it stopped describing the
system. That is not hypothetical here: the 2026-08-04 baseline (balanced
0.766) was measured on the WORKER route, and since 2026-08-06 production's
cheap judge rides the CRITIC branch — so the number in the file and the
system in production were never the same thing, and nothing said so.

THE FIX is not "re-bench more often". It is to make staleness *computable*,
so a re-bench happens exactly when one is owed and never otherwise:

    Tier 0  this script        ~0s, no LLM   — do I owe a re-bench at all?
    Tier 1  --cache-mode read  minutes       — replay; pay only for what
                                               genuinely changed
    Tier 2  --cache-mode off   full cost     — a fresh "number today"

A bench number is valid only while every input that can move it is
unchanged. This compares the CURRENT tree against the baseline's recorded
provenance and names each drifted component, with the cheapest action that
restores validity. Components, and why each is load-bearing:

  code.verifier   the system under test. Semantic AST digest, so comments
                  and docstring edits do NOT invalidate a benchmark but any
                  logic change does. This is the component that was MISSING
                  before 2026-08-09: templates and flags were fingerprinted,
                  the code was not, so a change to `_escalate_refute` moved
                  every number while provenance stayed byte-identical.
  code.bench      the ruler. Kept separate on purpose: the system changing
                  is a result, the ruler changing is not.
  templates.*     the prompts actually rendered (tuned artifacts included).
  cases/faults    the measurement set.
  judge/leg/arm   WHICH models and WHICH production rung. A leg change
                  invalidates everything — a different timeout ceiling and
                  an extra fallback rung is a different pipeline.
  verify_flags    the discipline switches that select the pipeline.

⚠ UNKNOWN IS NOT CHANGED. A baseline recorded before a field existed cannot
be compared on it. The first version of this script counted those as drift
and reported 15 stale components where ~2 were real — a checker that cries
wolf gets ignored, which is worse than no checker. Fields absent from the
baseline are reported separately as UNCOMPARABLE and never as drift.

Pass the SAME pool and topology flags you would pass to verify_bench.py;
otherwise the judge/leg/pool comparisons have nothing to compare against
and say so rather than guessing.

Exit codes: 0 = still valid, 1 = stale (something drifted), 2 = no baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ghost_agent.eval.verify_bench import (  # noqa: E402
    ARM_ESCALATED,
    build_case_pool,
    ARM_RAW,
    bench_provenance,
    load_cases_jsonl,
)

DEFAULT_BASELINE = "system/eval/verifier_incumbent_baseline.json"

# What each drifted component costs to restore. The whole point of the
# oracle is that most drift does NOT require the expensive path.
_REPLAYABLE = "replay (--cache-mode read): only changed prompts cost calls"
_FULL = "FULL live re-bench: cached responses cannot apply"

_COMPONENTS = {
    "code.verifier":  ("verifier logic changed (system under test)", _REPLAYABLE),
    "code.bench":     ("bench/scoring changed (the ruler, not the system)", _REPLAYABLE),
    "templates":      ("rendered prompt changed", _REPLAYABLE),
    "cases_sha256":   ("case pool changed", _REPLAYABLE),
    "faults_sha256":  ("fault library changed", _REPLAYABLE),
    "verify_flags":   ("a discipline switch selects a different pipeline", _REPLAYABLE),
    "judge":          ("judge endpoint/model changed", _FULL),
    "escalation.arm": ("measured pipeline changed (raw vs escalated)", _FULL),
    "escalation.leg": ("production rung changed (worker vs critic)", _FULL),
}


def _wilson_half_width(p: float, n: int, z: float = 1.96) -> float:
    if not n:
        return float("nan")
    return z * math.sqrt(max(p * (1.0 - p), 0.0) / n)


def honest_interval(base: dict) -> str:
    """The baseline's REAL uncertainty.

    ⚠ `smallest_resolvable_delta` in the baseline file is 0.5/min(class n) —
    the effect of ONE flipped trial, i.e. quantization, NOT statistical
    resolution. Read as power it understates the 95% half-width by ~6x, and
    a 0.0093 "resolution" invites shipping changes that are pure noise. The
    honest figure is computed here from the class sizes.
    """
    mix = base.get("class_mix") or {}
    n_nr, n_rf = mix.get("non_refute") or 0, mix.get("refute_expecting") or 0
    p_nr, p_rf = base.get("nonrefute_mean"), base.get("refute_mean")
    bal = base.get("private_incumbent_balanced")
    if None in (p_nr, p_rf, bal) or not (n_nr and n_rf):
        return ""
    half = 0.5 * math.sqrt(_wilson_half_width(p_nr, n_nr) ** 2
                           + _wilson_half_width(p_rf, n_rf) ** 2)
    quoted = base.get("smallest_resolvable_delta")
    extra = ""
    if quoted:
        extra = (f"\n  ⚠ the file's 'smallest_resolvable_delta' {quoted} is "
                 f"QUANTIZATION (0.5/min class n), not power — it understates "
                 f"this by {half / quoted:.1f}x")
    return (f"balanced {bal:.3f}  95% CI [{bal - half:.3f}, {bal + half:.3f}] "
            f"(±{half:.3f}, n={n_nr} non-refute / {n_rf} refute){extra}")


def flat_fingerprint(prov: dict) -> dict:
    """The comparable fingerprint, flattened to dotted keys."""
    esc = prov.get("escalation") or {}
    out = {
        "cases_sha256": prov.get("cases_sha256"),
        "faults_sha256": prov.get("faults_sha256"),
        "code.verifier": (prov.get("code") or {}).get("verifier"),
        "code.bench": (prov.get("code") or {}).get("bench"),
        "escalation.arm": esc.get("arm"),
        # The leg lives in two places across baseline versions; prefer the
        # explicit one and fall back rather than reporting "unknown" for a
        # run that did record it.
        "escalation.leg": (esc.get("cheap_route")
                           or (esc.get("route_health") or {}).get("leg")),
        "judge": json.dumps(prov.get("judge") or {}, sort_keys=True)
                 if prov.get("judge") else None,
    }
    for name, blk in (prov.get("templates") or {}).items():
        if isinstance(blk, dict):
            out[f"templates.{name}"] = blk.get("sha256")
    for k, v in (prov.get("verify_flags") or {}).items():
        out[f"verify_flags.{k}"] = v
    return out


def build_pool(seed_path: str, mined_path: Optional[str], no_mined: bool,
               tier: str):
    """Delegates to the bench's OWN pool builder — no replica.

    A hand-written copy here computed 35 cases where the bench loaded 58
    (it filtered seed cases by tier, which the bench does not, and skipped
    the mined-vs-seed dedup). That made `cases_sha256` report permanent
    false drift — the wolf-crying this whole tool exists to avoid. The copy
    is gone; there is one implementation.
    """
    return build_case_pool(seed_path, mined_path, no_mined, tier)


def _kind(key: str) -> tuple:
    for prefix, meta in _COMPONENTS.items():
        if key == prefix or key.startswith(prefix + "."):
            return meta
    return ("unclassified component", _FULL)


def compare(old: dict, new: dict) -> tuple:
    """(drift, uncomparable). UNKNOWN is never counted as drift."""
    drift, unknown = [], []
    for key in sorted(set(old) | set(new)):
        o, n = old.get(key), new.get(key)
        if o is None or n is None:
            # One side never recorded this. That is a gap in evidence, not
            # evidence of change, and conflating the two is what made the
            # first version of this tool useless.
            if o != n:
                unknown.append({
                    "component": key,
                    "missing_from": "baseline" if o is None else "current run",
                    "known_value": n if o is None else o,
                })
            continue
        if o != n:
            why, fix = _kind(key)
            drift.append({"component": key, "was": o, "now": n,
                          "why": why, "restore": fix})
    return drift, unknown


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", default="",
                    help=f"baseline JSON (default $GHOST_HOME/{DEFAULT_BASELINE})")
    ap.add_argument("--cases", default=str(REPO_ROOT / "scripts"
                                           / "verify_bench_cases.jsonl"))
    ap.add_argument("--mined-cases", default="")
    ap.add_argument("--no-mined", action="store_true")
    ap.add_argument("--tier", choices=("all", "public", "private"),
                    default="private",
                    help="must match the baseline's tier (the incumbent "
                         "baseline is recorded on the PRIVATE tier)")
    ap.add_argument("--base-url", default="",
                    help="judge endpoint you intend to run against; without "
                         "it the judge comparison is UNCOMPARABLE, not equal")
    ap.add_argument("--model", default="")
    ap.add_argument("--main-base-url", default="",
                    help="set => the escalated arm, matching production")
    ap.add_argument("--leg", choices=("critic", "worker"), default="critic")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    home = Path(os.environ.get("GHOST_HOME", ""))
    path = Path(args.baseline) if args.baseline else home / DEFAULT_BASELINE
    if not path.exists():
        print(f"NO BASELINE at {path} — nothing to compare; a first full run "
              f"is owed (--cache-mode write to seed the replay cache).")
        return 2
    base = json.loads(path.read_text())
    old = flat_fingerprint(base.get("provenance") or {})

    mined = args.mined_cases or (
        str(home / "system" / "eval" / "verify_bench_cases_mined.jsonl")
        if home.name else "")
    cases = build_pool(args.cases, mined, args.no_mined, args.tier)

    # Mirror what a RUN with these flags would record, so like compares to
    # like. Given no topology flags these stay None => UNCOMPARABLE.
    judge = ({"base_url": args.base_url, "model": args.model}
             if args.base_url else None)
    esc = None
    if args.base_url:
        esc = {"arm": ARM_ESCALATED if args.main_base_url else ARM_RAW,
               "cheap_route": args.leg}
    new = flat_fingerprint(bench_provenance(cases, judge=judge,
                                            escalation=esc))

    drift, unknown = compare(old, new)
    needs_full = any(d["restore"] == _FULL for d in drift)
    verdict = ("VALID" if not drift
               else "STALE — full live re-bench required" if needs_full
               else "STALE — replayable")

    if args.json:
        print(json.dumps({"verdict": verdict, "drift": drift,
                          "uncomparable": unknown, "n_cases": len(cases),
                          "baseline": str(path)}, indent=1))
        return 0 if not drift else 1

    print("=" * 78)
    print(f"VERIFIER BENCH STATUS   baseline recorded "
          f"{base.get('recorded_utc', '?')}")
    print("=" * 78)
    iv = honest_interval(base)
    if iv:
        print(f"  {iv}")
    print(f"  pool now: {len(cases)} cases (tier={args.tier})   "
          f"baseline: {base.get('n_private_cases', '?')}")
    print()

    if drift:
        print(f"  {verdict}   ({len(drift)} component(s) drifted)\n")
        for d in drift:
            print(f"  • {d['component']}  — {d['why']}")
            print(f"      was {str(d['was'])[:52]!r}")
            print(f"      now {str(d['now'])[:52]!r}")
            print(f"      -> {d['restore']}")
        print()
    else:
        print("  NO DRIFT in any comparable component.\n")

    if unknown:
        print(f"  UNCOMPARABLE ({len(unknown)}) — evidence missing, NOT "
              f"evidence of change:")
        for u in unknown:
            print(f"      {u['component']:<42} absent from {u['missing_from']}")
        print()

    if needs_full:
        print("  A cached replay CANNOT restore validity: the judge, arm or "
              "route changed,\n  so every cached response answers a different "
              "question. Run a fresh bench.")
    elif drift:
        print("  All drift is replayable. `--cache-mode read` pays for only "
              "the genuinely\n  changed prompts and reuses the rest.")
    else:
        print("  The recorded figure still describes the current tree.")
    print("=" * 78)
    return 1 if drift else 0


if __name__ == "__main__":
    sys.exit(main())
