#!/usr/bin/env python3
"""§4K Phase 2 — does the foresight shadow world model actually predict?

The gate every prospective signal here has to pass (the depth curve did,
router confidence is queued for it): bucket the resolved-prediction
ledger by predicted p(fail) and measure the ACTUAL failure rate in each.

Two outcomes, and they lead different places:

  * it discriminates  → the environment is predictable enough for a
    consumer; §4K Phase 3 (ONE consumer, as an experiment arm) may be
    designed — never enabled by this script;
  * it is flat        → tool outcomes here are not predictable from
    (tool, op, target) precedent at the counting floor. **The plan
    stops.** That is a result, not a failure — this project's history
    is full of signals that looked usable and measured dead.

Also reported, because a "discriminates" verdict is not one number:
per-basis accuracy (does exact-precedent beat class-precedent?),
coverage (how often the index could claim a probability at all), and
the Brier score of the claimed probabilities against outcomes.

Two instruments, one verdict machinery:

* **Live ledger** (default) — reads
  `$GHOST_HOME/system/foresight/predictions.jsonl`, which the shadow
  hook fills as production runs. The confirmatory instrument.
* **Offline replay** (`--offline-replay`) — leave-future-out replay of
  the EXISTING trajectory corpus: walk `user_request` turns
  chronologically, predict each tool call BEFORE observing it, grade
  against the recorded outcome, then feed the observation to the index.
  Available the day the module lands, so the Phase-2 question does not
  have to wait a week of accrual. Caveats it prints and you must carry:
  consecutive identical calls inside one turn are not independent
  (intervals are optimistic), the corpus spans prompt/label eras, and
  labels come from the shared corpus sniffer rather than the dispatch
  verdict — the same rule seeding uses, but not bit-identical to live.

Usage:
    PYTHONPATH=src python scripts/foresight_backtest.py
    PYTHONPATH=src python scripts/foresight_backtest.py --offline-replay
    PYTHONPATH=src python scripts/foresight_backtest.py --json
    PYTHONPATH=src python scripts/foresight_backtest.py --min-per-bucket 40

Exit codes: 0 = discriminates; 1 = flat or insufficient data; 2 = no data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ghost_agent.core.experiments import asymp_cs_radius  # noqa: E402

# Predicted-p(fail) buckets. 0.5 is kept as an edge — it is the match
# threshold the ledger's own `match` field uses, so if the signal is real
# the split should be visible right there. The lowest bucket is where a
# veto-style consumer would NOT fire; the top two are where it would.
BUCKETS = ((0.0, 0.15), (0.15, 0.35), (0.35, 0.5), (0.5, 0.75), (0.75, 1.01))

# A bucket below this many resolved rows is reported but never used for
# the verdict — the "no verdict on thin data" discipline.
DEFAULT_MIN_PER_BUCKET = 30

# Spread (worst minus best bucket actual-failure rate) below which the
# signal is called flat regardless of significance. Same bar as the
# router backtest: under 10 points is not worth a behaviour change.
DISCRIMINATION_THRESHOLD = 0.10

# Point estimates alone would false-fire on flat ground truth at small n
# (measured in the router backtest's header table), so the best and
# worst buckets' confidence-sequence intervals must ALSO be disjoint.
ALPHA = 0.05


def _default_ledger() -> Path:
    base = os.getenv("GHOST_HOME", "")
    return Path(base) / "system" / "foresight" / "predictions.jsonl"


def _bucket_of(p: float):
    for lo, hi in BUCKETS:
        if lo <= p < hi:
            return (lo, hi)
    return None


def iter_ledger_rows(path: Path):
    """Yield parsed rows from the ledger + its one rotation generation
    (oldest first so time-ordered analyses stay honest)."""
    for p in (Path(str(path) + ".1"), path):
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        except OSError:
            continue


def iter_replay_rows(trajectories_root: Path = None):
    """Leave-future-out replay: yield ledger-shaped rows by walking the
    trajectory corpus chronologically with a COLD index — every call is
    predicted strictly from calls that preceded it, then becomes an
    observation. No ledger is read or written; the corpus is read-only.

    Seeding is forced off instance-locally (`_seed_state="disabled"` —
    the replay IS the seed, applied in order; no env mutation, so a
    long-lived importer cannot have seeding silently disabled
    process-wide) and the singleton is not touched — a fresh index, so
    a live process and this script can never contaminate each other.

    The walk is restricted to `YYYY-MM-DD` day partitions in ascending
    order — the same guard seeding has, so a stray `archive/` dir can
    never replay as "newest" and break leave-future-out."""
    from ghost_agent.core.foresight import (
        _DAY_DIR_RE, Foresight, _normalize_error_head, call_target,
        is_synthetic_result, offline_call_failed)
    from ghost_agent.distill.collector import TrajectoryCollector

    inst = Foresight()
    inst._armed_logged = True          # CLI: no pretty_log banner
    inst._seed_state = "disabled"
    kwargs = {}
    if trajectories_root is not None:
        kwargs["root"] = trajectories_root
    collector = TrajectoryCollector(**kwargs)
    if not collector.root.exists():
        return
    days = sorted(p.name for p in collector.root.iterdir()
                  if p.is_dir() and _DAY_DIR_RE.match(p.name))

    def _walk():
        for day in days:
            yield from collector.iter_trajectories(day=day)

    for traj in _walk():
        if str(getattr(traj, "task_kind", "") or "") != "user_request":
            continue
        for tc in getattr(traj, "tool_calls", []) or []:
            tool = str(getattr(tc, "name", "") or "")[:64]
            if not tool:
                continue
            # Same population the live hook grades: pipeline-minted
            # rejections never executed, so they are not transitions.
            if is_synthetic_result(getattr(tc, "result", "")):
                continue
            args = getattr(tc, "arguments", None)
            args = args if isinstance(args, dict) else {}
            op = str(args.get("operation") or args.get("action") or "")[:64]
            target = call_target(tool, op, args)
            ok = not offline_call_failed(tc)
            pred = inst.predict(tool=tool, operation=op, target=target)
            if pred is None:            # kill switch — nothing to measure
                return
            # `tclass` is carried so the offline instrument buckets by the
            # SAME key as the live ledger (§4CL I0: core/imagination.py
            # keys its per-bucket gate on `(tool, tclass)`). Without it
            # the replay could only be bucketed by tool, which merges
            # `file_system` on a .py file with `file_system` on a URL —
            # a coarser key hides exactly the sub-buckets a gate is for.
            # `collect()` ignores the field; nothing else changes.
            # `fails` and `pred_err` join `tclass` for the same reason:
            # `core/imagination.is_steerable_row` defines the population
            # a steer would ACT on from exactly these fields, and an
            # offline instrument that cannot express that population
            # reports a different statistic than the live one under the
            # same name. Mirrors `foresight._write_ledger`'s row.
            row = {"tool": tool, "op": op, "tclass": pred.tclass,
                   "basis": pred.basis,
                   "support": pred.support, "fails": pred.fails,
                   "ok": ok}
            if pred.predicted_error:
                row["pred_err"] = pred.predicted_error[:200]
            if pred.p_fail is not None:
                row["p_fail"] = pred.p_fail
                row["match"] = (pred.p_fail >= 0.5) == (not ok)
            yield row
            # ⚠ The ERROR HEAD, not just the outcome. Seeding records it
            # (`foresight._seed_from_trajectories`) and the live hook
            # records it (`resolve`); this replay dropped it, so
            # `predicted_error` was empty on every offline prediction —
            # and `is_steerable_row` requires a non-empty error head,
            # because a claim with nothing to tell the model cannot be
            # acted on. The offline instrument was therefore reporting
            # ZERO steerable rows on a corpus that has thousands, which
            # reads as "no signal" rather than as "the instrument cannot
            # see it".
            err = ""
            if not ok:
                err = _normalize_error_head(
                    getattr(tc, "error", "") or getattr(tc, "result", ""))
            inst.observe(pred.tool, pred.op, pred.tclass, pred.sig, ok, err)


def collect(rows):
    """→ (per_bucket stats, per_basis stats, coverage counters, brier)."""
    stats = defaultdict(lambda: {"n": 0, "failed": 0, "outcomes": []})
    basis = defaultdict(lambda: {"n": 0, "matched": 0})
    cov = {"rows": 0, "claimed": 0, "unclaimed": 0, "out_of_range": 0}
    brier_sum = 0.0
    for rec in rows:
        try:
            cov["rows"] += 1
            p_fail = rec.get("p_fail")
            b = str(rec.get("basis") or "?")
            failed = not rec.get("ok", True)
            if p_fail is None:
                cov["unclaimed"] += 1
                continue
            cov["claimed"] += 1
            p_fail = float(p_fail)
            brier_sum += (p_fail - (1.0 if failed else 0.0)) ** 2
            basis[b]["n"] += 1
            if rec.get("match"):
                basis[b]["matched"] += 1
            key = _bucket_of(p_fail)
            if key is None:
                cov["out_of_range"] += 1
                continue
            s = stats[key]
            s["n"] += 1
            if failed:
                s["failed"] += 1
            s["outcomes"].append(1.0 if failed else 0.0)
        except Exception:
            continue
    brier = round(brier_sum / cov["claimed"], 4) if cov["claimed"] else None
    return stats, basis, cov, brier


REGRET_FILENAME = "regret.jsonl"

# ── The consistency ratio's own power ────────────────────────────────
# Pre-registered bar 0.70 against a null of 0.50 (the estimator below is
# conditional on exactly one branch passing, so the null IS a coin flip).
# Exact binomial power, two-sided alpha 0.05:
#
#     n discordant   10     20     30     49     80
#     power         0.149  0.416  0.589  0.803  0.941
#
# 49 is the smallest n reaching 80%. Via this project's own anytime-valid
# instrument (interval must exclude 0.50) it takes ~80. Below the floor
# the script refuses a verdict rather than printing one: a ratio of 1.000
# over a single pair is the §4CE "verdict without power" shape, and it
# would be sitting inside the very feature whose gate module exists
# because "a subset of 1 with precision 1.00 is not evidence".
MIN_DISCORDANT = 49
CONSISTENCY_BAR = 0.70
CONSISTENCY_NULL = 0.50

# Exit codes. THREE states, not two: "the instrument is not wired yet"
# and "it is wired and measured nothing" are opposite facts, and a loop
# or a CI job polls the exit code, not the prose.
EXIT_CONSISTENCY_PASS = 0
EXIT_CONSISTENCY_FAIL = 1
EXIT_CONSISTENCY_UNARMED = 2
EXIT_CONSISTENCY_UNDERPOWERED = 3


def _regret_path() -> Path:
    return (Path(os.getenv("GHOST_HOME", "")) / "system" / "imagination"
            / REGRET_FILENAME)


def _consistency_report(as_json: bool = False) -> int:
    """The consistency ratio, or an honest statement that it has no data.

    ⚠ THIS INSTRUMENT IS UNARMED UNTIL §4CL I4 SHIPS. The regret ledger
    is written by the nightly block that re-executes the REJECTED branch
    of a past ranking and grades it; until that exists there is nothing
    to compute a ratio over. It reports that state explicitly rather than
    printing a number derived from an empty file — a 0.0 or a 1.00 from
    zero rows is exactly the shape of a measurement nobody can act on.

    Exit codes: 0 = ratio at or above the pre-registered 0.70 bar;
    1 = below it; 2 = no data (unarmed).
    """
    path = _regret_path()
    rows = []
    for p in (Path(str(path) + ".1"), path):
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        continue
        except OSError:
            continue

    graded = [r for r in rows
              if r.get("chosen_outcome") in ("pass", "fail")
              and r.get("rejected_outcome") in ("pass", "fail")]
    # Clustering: several regret pairs can come from ONE trajectory, and
    # those are not independent (errors compound — the whole reason this
    # is a trajectory-level metric). Reported so the dependence is
    # visible; the binomial interval below still assumes independence,
    # which makes it OPTIMISTIC when the clusters are large.
    clusters = len({str(r.get("req_id") or r.get("trajectory_id") or i)
                    for i, r in enumerate(graded)})
    if not graded:
        msg = ("consistency ratio: NO DATA — the regret ledger is empty "
               f"({path}). This instrument is armed by §4CL I4 (the nightly "
               "block that re-executes a rejected branch and grades it); "
               "until then Imagine's go/no-go number does not exist and "
               "must not be substituted for per-step accuracy.")
        if as_json:
            print(json.dumps({"instrument": "consistency", "n": 0,
                              "ratio": None, "armed": False,
                              "ledger": str(path)}))
        else:
            print(msg, file=sys.stderr)
        return EXIT_CONSISTENCY_UNARMED

    # "The predicted-better branch actually succeeded" — counted only on
    # DISCRIMINATING pairs. A pair where both branches passed (or both
    # failed) says nothing about the ranking and would dilute the ratio
    # toward whatever the base pass rate happens to be.
    disc = [r for r in graded
            if r["chosen_outcome"] != r["rejected_outcome"]]
    right = sum(1 for r in disc if r["chosen_outcome"] == "pass")
    ratio = (right / len(disc)) if disc else None
    by_bucket = defaultdict(lambda: [0, 0])
    for r in disc:
        b = f"{r.get('tool', '?')}|{r.get('tclass', '')}"
        by_bucket[b][1] += 1
        if r["chosen_outcome"] == "pass":
            by_bucket[b][0] += 1

    # The interval on the ratio, from the project's own instrument.
    radius = None
    if disc:
        try:
            radius = asymp_cs_radius([1.0 if r["chosen_outcome"] == "pass"
                                      else 0.0 for r in disc])
        except Exception:
            radius = None
    powered = len(disc) >= MIN_DISCORDANT
    verdict = ("UNDERPOWERED" if not powered else
               "PASS" if ratio is not None and ratio >= CONSISTENCY_BAR
               else "BELOW BAR")

    if as_json:
        print(json.dumps({
            "instrument": "consistency", "armed": True,
            "n_graded": len(graded), "n_discriminating": len(disc),
            "n_clusters": clusters,
            "min_discordant": MIN_DISCORDANT,
            "powered": powered, "verdict": verdict,
            "ratio": ratio, "ci_radius": radius,
            "bar": CONSISTENCY_BAR, "null": CONSISTENCY_NULL,
            "by_bucket": {k: {"right": v[0], "n": v[1]}
                          for k, v in by_bucket.items()},
        }))
    else:
        print("§4CL I0 — trajectory-level consistency ratio")
        # DISCORDANT first and largest: it is the effective sample size.
        # `graded` is bigger and means less, and leading with it is how a
        # reader mis-scales the bar.
        print(f"  discordant pairs (the two branches disagreed): "
              f"{len(disc)}   [the effective n]")
        print(f"  graded regret pairs: {len(graded)} "
              f"across {clusters} trajector(ies)")
        if ratio is None:
            print("  ratio: — (no pair where the branches disagreed. That "
                  "is a MEASUREMENT — the ranking never changed an "
                  "outcome — not an unwired instrument)")
            return EXIT_CONSISTENCY_UNDERPOWERED
        _ci = "" if radius is None else f" ±{radius:.3f}"
        print(f"  ratio: {ratio:.3f}{_ci}   bar {CONSISTENCY_BAR:.2f} "
              f"against a null of {CONSISTENCY_NULL:.2f}")
        if not powered:
            print(f"  VERDICT: UNDERPOWERED — {MIN_DISCORDANT} discordant "
                  f"pairs are needed for 80% power at this bar; "
                  f"{MIN_DISCORDANT - len(disc)} more to go. The point "
                  f"estimate above is not a result.")
        else:
            print(f"  VERDICT: {verdict}")
        for b, (right, n) in sorted(by_bucket.items(),
                                    key=lambda kv: -kv[1][1]):
            print(f"    {b:38} {right}/{n} = {right / n:.2f}")
    if ratio is None or not powered:
        return EXIT_CONSISTENCY_UNDERPOWERED
    return (EXIT_CONSISTENCY_PASS if ratio >= CONSISTENCY_BAR
            else EXIT_CONSISTENCY_FAIL)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ledger", default="")
    ap.add_argument("--offline-replay", action="store_true",
                    help="leave-future-out replay of the trajectory "
                         "corpus instead of the live ledger")
    ap.add_argument("--trajectories", default="",
                    help="trajectory root override (offline replay only)")
    ap.add_argument("--min-per-bucket", type=int,
                    default=DEFAULT_MIN_PER_BUCKET)
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--consistency", action="store_true",
        help="§4CL I0: trajectory-level CONSISTENCY RATIO instead of the "
             "bucket table. Over the pairs where the chosen and rejected "
             "branches DISAGREED, the fraction where the chosen one was "
             "the branch that worked — i.e. the McNemar discordant-pairs "
             "estimator, whose null is 0.50. (On the marginal scale the "
             "null would be the base pass rate ~0.90 and a 0.70 bar would "
             "mean WORSE than doing nothing; the conditional scale is the "
             "only one on which the pre-registered bar means anything.) "
             "Per-step accuracy is a lying metric for a planner because "
             "errors compound, so this is the phase's go/no-go number. "
             "Reads system/imagination/regret.jsonl, which only I4 writes.")
    args = ap.parse_args()

    if not args.ledger and not args.trajectories \
            and not os.getenv("GHOST_HOME", "").strip():
        print("GHOST_HOME is not set and no --ledger/--trajectories "
              "given — refusing to guess a relative path", file=sys.stderr)
        if args.json:
            print(json.dumps({"error": "no_ghost_home"}))
        return 2

    # ⚠ BELOW the GHOST_HOME guard, deliberately. `_regret_path` is
    # `Path(os.getenv("GHOST_HOME","")) / …`, which with the var unset is
    # a RELATIVE path — and a go/no-go verdict computed from a stray
    # `system/imagination/regret.jsonl` under the operator's cwd is
    # exactly what the guard above refuses to do for the ledger.
    if args.consistency:
        return _consistency_report(as_json=args.json)

    if args.offline_replay:
        troot = (Path(args.trajectories) if args.trajectories
                 else Path(os.getenv("GHOST_HOME", ""))
                 / "system" / "trajectories")
        if not troot.exists():
            print(f"no trajectory corpus at {troot}", file=sys.stderr)
            if args.json:
                print(json.dumps({"error": "corpus_missing",
                                  "root": str(troot)}))
            return 2
        source_rows = iter_replay_rows(
            Path(args.trajectories) if args.trajectories else None)
        instrument = "OFFLINE REPLAY (leave-future-out)"
    else:
        ledger = Path(args.ledger) if args.ledger else _default_ledger()
        if not ledger.exists() and not Path(str(ledger) + ".1").exists():
            print(f"no foresight ledger at {ledger}", file=sys.stderr)
            if args.json:
                print(json.dumps({"error": "ledger_missing",
                                  "ledger": str(ledger)}))
            return 2
        source_rows = iter_ledger_rows(ledger)
        instrument = "live ledger"

    stats, basis, cov, brier = collect(source_rows)
    bucket_alpha = ALPHA / max(1, len(BUCKETS))

    rows = []
    for key in BUCKETS:
        s = stats.get(key)
        if not s or not s["outcomes"]:
            rows.append({"bucket": f"{key[0]:.2f}-{key[1]:.2f}", "n": 0,
                         "failure_rate": None, "ci": None, "usable": False})
            continue
        n = len(s["outcomes"])
        rate = s["failed"] / n
        rad = asymp_cs_radius(s["outcomes"], alpha=bucket_alpha)
        rows.append({
            "bucket": f"{key[0]:.2f}-{key[1]:.2f}", "n": n,
            "failure_rate": round(rate, 4),
            "ci": None if rad is None else round(rad, 4),
            "usable": n >= args.min_per_bucket,
        })

    usable = [r for r in rows if r["usable"] and r["ci"] is not None]
    spread = None
    disjoint = False
    monotone = None
    verdict = "insufficient data"
    if len(usable) >= 2:
        best = min(usable, key=lambda r: r["failure_rate"])
        worst = max(usable, key=lambda r: r["failure_rate"])
        spread = round(worst["failure_rate"] - best["failure_rate"], 4)
        disjoint = ((best["failure_rate"] + best["ci"])
                    < (worst["failure_rate"] - worst["ci"]))
        # A real predictor should ALSO be ordered: higher predicted
        # p(fail) → higher actual failure. Reported, not required — two
        # disjoint extremes with a noisy middle is still a usable signal.
        rates = [r["failure_rate"] for r in usable]
        monotone = all(a <= b + 1e-9 for a, b in zip(rates, rates[1:]))
        if spread >= DISCRIMINATION_THRESHOLD and disjoint:
            verdict = "DISCRIMINATES"
        elif spread >= DISCRIMINATION_THRESHOLD:
            verdict = ("SPREAD BUT NOT SIGNIFICANT — best/worst intervals "
                       "overlap; collect more before reading anything in")
        else:
            verdict = ("FLAT — tool outcomes are not predictable from "
                       "(tool, op, target) precedent at the counting "
                       "floor; §4K stops at Phase 2")

    basis_rows = {
        b: {"n": v["n"],
            "accuracy": round(v["matched"] / v["n"], 3) if v["n"] else None}
        for b, v in sorted(basis.items())
    }

    if args.json:
        print(json.dumps({"instrument": instrument, "coverage": cov,
                          "brier": brier,
                          "buckets": rows, "by_basis": basis_rows,
                          "spread": spread, "intervals_disjoint": disjoint,
                          "monotone": monotone, "verdict": verdict},
                         indent=2))
    else:
        print("§4K Phase 2 — predicted p(fail) vs actual failure rate")
        print(f"  instrument: {instrument}")
        if args.offline_replay:
            print("  ⚠ replay caveats: within-turn calls are not "
                  "independent (intervals optimistic); corpus spans "
                  "label/prompt eras; labels are the corpus sniffer's, "
                  "not the live dispatch verdict. Confirm on the live "
                  "ledger before Phase 3 ships at full traffic.")
        print(f"  rows: {cov['rows']} graded predictions, "
              f"{cov['claimed']} claimed a probability "
              f"({cov['unclaimed']} no-precedent)")
        if cov["rows"] == 0:
            print("  ⚠ EMPTY source — nothing to measure yet.")
        print()
        print(f"  {'p(fail)':<14}{'n':>6}{'failure rate':>14}{'±CS':>9}")
        for r in rows:
            fr = ("—" if r["failure_rate"] is None
                  else f"{r['failure_rate']:.3f}")
            ci = "—" if r["ci"] is None else f"{r['ci']:.3f}"
            flag = "" if r["usable"] else "   (thin)"
            print(f"  {r['bucket']:<14}{r['n']:>6}{fr:>14}{ci:>9}{flag}")
        print()
        if brier is not None:
            print(f"  Brier of claimed probabilities: {brier} "
                  f"(coin at the base rate would need comparing — see "
                  f"the ledger's actual_fail_rate in learning-health)")
        print(f"  per-basis accuracy: " + ", ".join(
            f"{b}={v['accuracy']} (n={v['n']})"
            for b, v in basis_rows.items()) if basis_rows else "")
        if spread is not None:
            print(f"  spread across usable buckets: {spread:.3f} "
                  f"(threshold {DISCRIMINATION_THRESHOLD}); "
                  f"intervals disjoint: {disjoint}; "
                  f"ordered by prediction: {monotone}")
        print(f"  VERDICT: {verdict}")
        print()
    return 0 if verdict == "DISCRIMINATES" else 1


if __name__ == "__main__":
    raise SystemExit(main())
