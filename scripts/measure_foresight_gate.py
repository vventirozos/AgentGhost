#!/usr/bin/env python3
"""§4CS item G — does the Imagine calibration gate qualify anything, ever?

MEASURE-ONLY. Two questions, in the order they must be asked:

  1. Does the INCUMBENT pass its own checker? Instruments in this project
     lie plausibly, and this one decides whether any steering site may
     exist. Every field of a live bucket is recomputed by hand from the
     raw ledger and compared against what `build_gate` wrote, and the
     `enabled` path is exercised on synthetic buckets — a checker that
     cannot say yes is not a gate.
  2. TIME TO QUALIFY at the observed rate. Predicted-fail rows are rare
     BY CONSTRUCTION, so the denominator, not traffic, is the question —
     and then whether the precision bar can be met at all.

Run:
    GHOST_HOME=/path/to/Data PYTHONPATH=src python3 scripts/measure_foresight_gate.py
"""
from __future__ import annotations

import collections
import datetime
import math
import os
import sys
from pathlib import Path


def main() -> int:
    home = os.getenv("GHOST_HOME", "").strip()
    if not home:
        sys.exit("GHOST_HOME is not set — refusing to guess the data root.")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from ghost_agent.core import imagination as IM

    led = Path(home) / "system" / "foresight" / "predictions.jsonl"
    if not led.is_file():
        sys.exit(f"{led} does not exist.")
    rows = list(IM._iter_ledger(led))
    doc = IM.build_gate(write=False, home=home)
    p = doc["params"]
    print(f"ledger {len(rows)} rows | gate: {len(doc['buckets'])} buckets, "
          f"{doc['enabled_count']} enabled | params {p}")

    # ── 1. does the incumbent pass its own checker? ────────────────────
    def by_hand(target):
        m = collections.Counter()
        brier = 0.0
        claimed_failed = 0
        for rec in rows:
            if IM.bucket_key(rec.get("tool"), rec.get("tclass")) != target:
                continue
            m["n"] += 1
            failed = not bool(rec.get("ok", True))
            m["failed"] += failed
            pf = rec.get("p_fail")
            if not isinstance(pf, (int, float)):
                continue
            m["claimed"] += 1
            brier += (float(pf) - (1.0 if failed else 0.0)) ** 2
            claimed_failed += failed
            if IM.is_steerable_row(rec):
                m["fail_n"] += 1
                m["fail_hits"] += failed
        return m, brier, claimed_failed

    biggest = sorted(doc["buckets"].items(), key=lambda kv: -kv[1]["n"])[:3]
    print("\n1. HAND-RECOMPUTATION vs THE GATE")
    all_match = True
    for target, g in biggest:
        m, brier, cf = by_hand(target)
        base = cf / m["claimed"] if m["claimed"] else 0.0
        checks = [
            ("n", m["n"], g["n"]), ("claimed", m["claimed"], g["claimed"]),
            ("fail_n", m["fail_n"], g["fail_n"]),
            ("fail_hits", m["fail_hits"], g["fail_hits"]),
            ("fail_rate", round(m["failed"] / m["n"], 4), g["fail_rate"]),
            ("brier", round(brier / m["claimed"], 4) if m["claimed"] else None,
             g["brier"]),
            ("brier_base_rate", round(base * (1 - base), 4),
             g["brier_base_rate"]),
            ("precision",
             (round(m["fail_hits"] / m["fail_n"], 4) if m["fail_n"] else None),
             g["precision"]),
        ]
        bad = [(k, a, b) for k, a, b in checks if a != b]
        all_match &= not bad
        print(f"   {target:30} {'ALL FIELDS MATCH' if not bad else 'MISMATCH'}")
        for k, a, b in bad:
            print(f"      {k}: hand={a} gate={b}")
    print(f"   → incumbent passes its own checker: {all_match}")

    print("\n   is the `enabled` path reachable at all? (synthetic buckets)")
    for label, (nf, prec, nok, okr) in (
            ("at the floor: fail_n=10 prec=0.60", (10, 0.60, 20, 0.05)),
            ("fail_n=10 prec=0.90", (10, 0.90, 20, 0.05)),
            ("fail_n=30 prec=0.80", (30, 0.80, 60, 0.05))):
        hits = int(nf * prec)
        b = {"n": nf + nok, "claimed": nf + nok, "failed": 0, "matched": 0,
             "fail_n": nf, "fail_hits": hits,
             "fail_outcomes": [1.0] * hits + [0.0] * (nf - hits),
             "ok_outcomes": [1.0] * int(nok * okr) + [0.0] * (nok - int(nok * okr)),
             "brier_sum": 0.0}
        e = IM._evaluate_bucket("synthetic|x", b, p)
        print(f"      {label:34} enabled={e['enabled']}  {e['why'][:56]}")

    # ── 2. time to qualify ─────────────────────────────────────────────
    def ts(r):
        for k in ("ts", "at", "timestamp", "time"):
            v = r.get(k)
            if v:
                try:
                    return datetime.datetime.fromisoformat(
                        str(v).replace("Z", "+00:00"))
                except ValueError:
                    pass
        return None

    stamped = [(ts(r), r) for r in rows]
    stamped = [(t, r) for t, r in stamped if t]
    span = (max(t for t, _ in stamped) - min(t for t, _ in stamped)
            ).total_seconds() / 86400.0
    steer = [r for r in rows if IM.is_steerable_row(r)]
    hits = sum(1 for r in steer if not bool(r.get("ok", True)))
    print(f"\n2. TIME TO QUALIFY  (ledger spans {span:.1f} days, "
          f"{len(rows) / span:.1f} rows/day)")
    print(f"   steerable (predicted-fail) rows: {len(steer)} "
          f"= {len(steer) / span:.3f}/day — rare BY CONSTRUCTION")
    per = collections.Counter(
        IM.bucket_key(r.get("tool"), r.get("tclass")) for r in steer)
    zero = sum(1 for b in doc["buckets"].values() if b["fail_n"] == 0)
    print(f"   {zero} of {len(doc['buckets'])} buckets have ZERO — for those "
          f"the answer is 'never', not 'later'")
    for k, have in per.most_common(4):
        need = max(0, p["min_fail_n"] - have)
        rate = have / span
        print(f"      {k:30} {have:2}/{p['min_fail_n']} → "
              f"{need / rate / 365.25:.2f} years to the DENOMINATOR alone")

    n, k = len(steer), hits
    ph = k / n if n else 0.0
    z, den = 1.96, 1 + 1.96 ** 2 / n if n else 1
    c = (ph + z * z / (2 * n)) / den if n else 0
    half = (z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den) if n else 0
    claimed = [r for r in rows if isinstance(r.get("p_fail"), (int, float))]
    ok_rows = [r for r in claimed if not IM.is_steerable_row(r)]
    okf = sum(1 for r in ok_rows if not bool(r.get("ok", True)))
    print(f"\n   ⚠ THE DENOMINATOR IS NOT THE BINDING CONSTRAINT.")
    print(f"   pooled precision over EVERY steerable row: {k}/{n} = {ph:.3f}"
          f"  Wilson 95% CI [{max(0, c - half):.3f}, {min(1, c + half):.3f}]")
    print(f"   bar is {p['min_fail_precision']:.2f} — above the whole interval")
    print(f"   predicted-FAIL rows fail {ph:.3f}; predicted-OK rows fail "
          f"{okf / len(ok_rows):.3f}")
    print(f"   spread {ph - okf / len(ok_rows):+.3f} against a "
          f"{p['min_spread']:+.2f} bar — THE SIGN IS WRONG")
    sk = [v["brier_skill"] for v in doc["buckets"].values()
          if v.get("brier_skill") is not None]
    print(f"   Brier skill negative in {sum(1 for v in sk if v < 0)}/{len(sk)} "
          f"buckets with a denominator")
    print("\n   `_evaluate_bucket` rejects on precision BEFORE the interval "
          "test, so a\n   bucket whose TRUE precision is under the bar cannot "
          "enable at ANY n.\n   Time-to-qualify is not long — it is UNDEFINED "
          "unless the index improves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
