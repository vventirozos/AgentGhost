#!/usr/bin/env python3
"""Compare two verifier bench runs — PAIRED, and refusing to compare apples.

WHY PAIRED. The absolute balanced score carries a 95% half-width of about
±0.05 at this bench's size, so comparing two runs as independent proportions
can only resolve changes bigger than ~0.1 — larger than almost any real
improvement. But the two runs score the SAME cases under the SAME faults, so
the trials pair up, and a paired test looks only at the trials that actually
CHANGED. Ten flips one way and two the other is decisive evidence even though
both runs' absolute intervals overlap almost entirely.

That is McNemar's test: given b trials that went right->wrong and c that went
wrong->right, the null is b ~ Binomial(b+c, 0.5). Concordant pairs carry no
information about the difference and are correctly ignored. The exact
two-sided binomial p-value is computed here (no chi-square approximation,
which is unreliable at the small discordant counts this bench produces).

WHY IT REFUSES. The 2026-08-04 baseline was measured on the WORKER route and
production moved to CRITIC two days later; the two numbers were compared
anyway because nothing checked. This tool diffs the provenance fingerprints
FIRST and refuses outright when the runs measured different systems — unless
you name the difference with --expect-differs, which is how you legitimately
A/B one component. A comparison you cannot justify is not a comparison.

    PYTHONPATH=src python scripts/verify_bench_compare.py OLD.json NEW.json
    ... --expect-differs code.verifier      # A/B of a verifier change
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from verify_bench_status import compare as fingerprint_compare  # noqa: E402
from verify_bench_status import flat_fingerprint  # noqa: E402

# Expectation per fault class: what a CORRECT verifier does with this trial.
# Kept here rather than re-derived so the comparison scores the same notion
# of "right" the bench itself does.
_NON_REFUTE = {"clean", "evidence_truncation"}


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p for b successes in b+c at p=0.5.

    Exact rather than chi-square: with the discordant counts this bench
    produces (often < 20) the approximation is unreliable in exactly the
    regime where the answer matters.
    """
    n = b + c
    if n == 0:
        return 1.0
    def _pmf(k):
        return math.comb(n, k) * (0.5 ** n)
    obs = _pmf(b)
    # Sum every outcome at most as likely as the observed one; the 1e-12
    # slack keeps floating-point ties (symmetric k) from being dropped.
    return min(1.0, sum(_pmf(k) for k in range(n + 1) if _pmf(k) <= obs * (1 + 1e-12)))


def _correct(trial: dict) -> bool:
    """Did this trial land on the verdict its fault class calls for?"""
    v, fault = trial.get("verdict"), trial.get("fault")
    if v is None:
        return False
    return (v != "REFUTED") if fault in _NON_REFUTE else (v == "REFUTED")


def _trials(report: dict, arm: str) -> dict:
    arms = report.get("arms") or {}
    if arm not in arms:
        raise SystemExit(f"arm {arm!r} not in report (have: {sorted(arms)})")
    out = {}
    for t in arms[arm].get("trials") or []:
        out[(t.get("case_id"), t.get("fault"))] = t
    return out


def _rate(trials, keys, want_non_refute=None):
    sel = [trials[k] for k in keys
           if want_non_refute is None
           or ((trials[k].get("fault") in _NON_REFUTE) == want_non_refute)]
    if not sel:
        return None, 0
    return sum(_correct(t) for t in sel) / len(sel), len(sel)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old"); ap.add_argument("new")
    ap.add_argument("--arm", default="two_stage_on")
    ap.add_argument("--expect-differs", action="append", default=[],
                    help="fingerprint component the two runs are SUPPOSED to "
                         "differ in (the thing under test). Repeatable.")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    a = json.loads(Path(args.old).read_text())
    b = json.loads(Path(args.new).read_text())

    # ── comparability gate, BEFORE any number is computed ──────────────
    drift, unknown = fingerprint_compare(
        flat_fingerprint(a.get("provenance") or {}),
        flat_fingerprint(b.get("provenance") or {}))
    unexpected = [d for d in drift if d["component"] not in args.expect_differs]

    lines = []
    if unexpected:
        lines.append("REFUSING TO COMPARE — the runs measured different "
                     "systems in ways you did not declare:")
        for d in unexpected:
            lines.append(f"  • {d['component']}: {str(d['was'])[:28]!r} -> "
                         f"{str(d['now'])[:28]!r}   ({d['why']})")
        lines.append("")
        lines.append("  Declare the intended difference with --expect-differs "
                     "<component>, or\n  re-run both sides under the same "
                     "configuration.")
        print("\n".join(lines))
        return 2

    # Replay honesty: a replayed side measures code against a frozen judge.
    modes = [(r.get("provenance", {}).get("cache") or {}).get("measures", "?")
             for r in (a, b)]

    ta, tb = _trials(a, args.arm), _trials(b, args.arm)
    keys = sorted(set(ta) & set(tb))
    if not keys:
        print("no trials pair up between these runs (disjoint case sets)")
        return 2

    # ── the paired test ────────────────────────────────────────────────
    b_cnt = sum(1 for k in keys if _correct(ta[k]) and not _correct(tb[k]))
    c_cnt = sum(1 for k in keys if not _correct(ta[k]) and _correct(tb[k]))
    p = mcnemar_exact(b_cnt, c_cnt)

    old_all, n_all = _rate(ta, keys)
    new_all, _ = _rate(tb, keys)
    old_nr, n_nr = _rate(ta, keys, want_non_refute=True)
    new_nr, _ = _rate(tb, keys, want_non_refute=True)
    old_rf, n_rf = _rate(ta, keys, want_non_refute=False)
    new_rf, _ = _rate(tb, keys, want_non_refute=False)
    old_bal = (0.5 * (old_nr + old_rf)) if None not in (old_nr, old_rf) else None
    new_bal = (0.5 * (new_nr + new_rf)) if None not in (new_nr, new_rf) else None

    decided = p < args.alpha
    direction = ("NEW BETTER" if c_cnt > b_cnt else
                 "NEW WORSE" if b_cnt > c_cnt else "no direction")
    verdict = (f"{direction} (p={p:.4f} < {args.alpha})" if decided
               else f"NO DIFFERENCE RESOLVED (p={p:.4f})")

    if args.json:
        print(json.dumps({
            "verdict": verdict, "p": p, "decided": decided,
            "regressed": b_cnt, "improved": c_cnt, "paired_trials": len(keys),
            "old_balanced": old_bal, "new_balanced": new_bal,
            "cache_modes": modes,
            "uncomparable_components": [u["component"] for u in unknown],
        }, indent=1))
        return 0

    print("=" * 78)
    print(f"PAIRED BENCH COMPARISON   arm={args.arm}   {len(keys)} paired trials")
    print("=" * 78)
    for lbl, r, m in (("OLD", a, modes[0]), ("NEW", b, modes[1])):
        print(f"  {lbl}: {r.get('started_utc', r.get('recorded_utc', '?'))}   [{m}]")
    if any("replayed" in m for m in modes):
        print("  ⚠ a replayed side holds the judge FIXED — this measures the "
              "CODE, which is\n    what you want for attribution and is NOT a "
              "statement about current quality.")
    if unknown:
        print(f"  ({len(unknown)} component(s) uncomparable — evidence "
              f"missing, not evidence of change)")
    print()
    # A class can be empty (a fault-subset run, or a pool with no clean
    # cases). Say so rather than crashing on the format — and never print a
    # "balanced" score that only one of its two halves actually supports.
    def _row(label, o, n, cnt):
        if o is None or n is None:
            print(f"  {label:<13} n/a — no trials in this class")
        else:
            print(f"  {label:<13} {o:.4f} -> {n:.4f}   "
                  f"(delta {n - o:+.4f}, n={cnt})")
    if old_bal is None or new_bal is None:
        print("  balanced      NOT COMPUTED — one class has no trials, so a "
              "balanced score\n                would be a raw score wearing "
              "a balanced label")
    else:
        _row("balanced", old_bal, new_bal, n_nr + n_rf)
    _row("  non-refute", old_nr, new_nr, n_nr)
    _row("  refute", old_rf, new_rf, n_rf)
    _row("raw accuracy", old_all, new_all, n_all)
    print()
    print("  McNemar (paired, exact binomial) — concordant trials carry no")
    print("  information about a difference and are correctly ignored:")
    print(f"    right -> wrong : {b_cnt}")
    print(f"    wrong -> right : {c_cnt}")
    print(f"    discordant     : {b_cnt + c_cnt} of {len(keys)}")
    print(f"\n  VERDICT: {verdict}")
    if not decided and (b_cnt + c_cnt) < 6:
        print(f"  Too few trials CHANGED ({b_cnt + c_cnt}) to resolve anything; "
              f"this is a\n  power limit, not evidence the change is inert.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
