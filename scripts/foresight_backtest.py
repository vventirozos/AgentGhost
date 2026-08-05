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
        _DAY_DIR_RE, Foresight, call_target, is_synthetic_result,
        offline_call_failed)
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
            row = {"tool": tool, "op": op, "basis": pred.basis,
                   "support": pred.support, "ok": ok}
            if pred.p_fail is not None:
                row["p_fail"] = pred.p_fail
                row["match"] = (pred.p_fail >= 0.5) == (not ok)
            yield row
            inst.observe(pred.tool, pred.op, pred.tclass, pred.sig, ok)


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
    args = ap.parse_args()

    if not args.ledger and not args.trajectories \
            and not os.getenv("GHOST_HOME", "").strip():
        print("GHOST_HOME is not set and no --ledger/--trajectories "
              "given — refusing to guess a relative path", file=sys.stderr)
        if args.json:
            print(json.dumps({"error": "no_ghost_home"}))
        return 2

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
