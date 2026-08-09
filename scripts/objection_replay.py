#!/usr/bin/env python3
"""Replay-scorer for mechanical-layer experiments — seconds, not hours.

THE POINT (2026-08-07): the cheap judge's verdicts are UPSTREAM of
everything the escalation policies iterate on, so a policy change does
not need 3 hours of re-inference — it needs the recorded verdicts plus a
live top-up for exactly the trials whose escalation population changed.

Two modes:

  splice  — take a recorded bundle, deterministically REGENERATE its
            trial list (per-case seeding makes this exact), select the
            trials a flag predicate marks (default: objection_dismissed),
            re-run ONLY those live under the CURRENT env, splice the new
            results over the recorded ones, and score both variants
            side by side. Cost: N_selected live verifies (~30s each).

            PYTHONPATH=src python scripts/objection_replay.py splice \
                --bundle verify_bench_out/20260807T102629Z \
                --base-url http://100.83.184.117:8088 \
                --main-base-url http://127.0.0.1:8088

  rescore — pure offline: for bundles whose trials carry the
            `cheap_verdict` snapshot (recorded from 2026-08-07 on),
            re-run the mechanical layer alone over the recorded cheap
            verdicts and report which trials WOULD change disposition
            under the current code/flags. Zero inference; trials whose
            new disposition needs an escalation outcome the recording
            lacks are listed as UNKNOWN rather than guessed.

            PYTHONPATH=src python scripts/objection_replay.py rescore \
                --bundle verify_bench_out/<ts>

The splice mode trusts two invariants, both checked before any call is
spent: (1) the regenerated (case_id, fault) sequence matches the
bundle's exactly — per-case seeding is what makes that hold; (2) the
case pool hash matches the bundle's provenance. A mismatch aborts: a
splice over a drifted pool would silently compare two different
benchmarks, the exact defect class the provenance block exists to stop.

Cheap-judge nondeterminism is REPORTED, not hidden: every re-run trial's
fresh cheap verdict is compared against the recorded one and drift is
printed (temperature 0.1 keeps it rare, not impossible).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ghost_agent.core.verifier import Verifier               # noqa: E402
from ghost_agent.core import objection                       # noqa: E402
from ghost_agent.eval import verify_bench as vb              # noqa: E402


def _load_bundle(bundle_dir: Path):
    data = json.loads((bundle_dir / "results.json").read_text())
    arms = data.get("arms") or {}
    if len(arms) != 1:
        raise SystemExit(f"expected exactly one arm in {bundle_dir}, "
                         f"got {list(arms)}")
    arm_name, arm = next(iter(arms.items()))
    return data, arm_name, arm["trials"]


def _pool_from_provenance(data) -> list:
    """Rebuild the exact case pool the bundle ran on: checked-in seeds +
    the persistent mined pool filtered to the recorded tier."""
    cases = vb.load_cases_jsonl(
        Path(__file__).resolve().parent / "verify_bench_cases.jsonl")
    ghost_home = os.getenv("GHOST_HOME", "")
    mined = (Path(ghost_home) / "system" / "eval"
             / "verify_bench_cases_mined.jsonl")
    if mined.exists():
        extra = vb.load_cases_jsonl(mined)
        have = {(c.claim.strip(), c.evidence.strip()) for c in cases}
        extra = [c for c in extra
                 if (c.claim.strip(), c.evidence.strip()) not in have]
        from ghost_agent.optim.trainset import holdout_tier
        tier = "private"      # the only tier the gate runs on
        extra = [c for c in extra
                 if holdout_tier(f"vbcase:{c.case_id}",
                                 private_pct=30) == tier]
        cases.extend(extra)
    # Same identity digest the harness records.
    dig = hashlib.sha256()
    for c in sorted(cases, key=lambda x: (x.case_id, x.claim, x.evidence,
                                          x.context)):
        dig.update((c.case_id + "\x00" + c.claim + "\x00" + c.evidence
                    + "\x00" + c.context + "\x00").encode("utf-8"))
    sha = dig.hexdigest()[:16]
    want = (data.get("provenance") or {}).get("cases_sha256")
    if want and sha != want:
        raise SystemExit(
            f"case-pool drift: regenerated sha {sha} != bundle sha {want}. "
            f"A splice over a drifted pool compares two different "
            f"benchmarks — refusing.")
    return cases


def _balanced(trials_rows) -> float:
    nr, rf = [], []
    for t in trials_rows:
        v, e = t["verdict"], t["expected"]
        if v is None:
            continue
        if e == "NOT_REFUTED":
            s = 1.0 if v in ("CONFIRMED", "UNCERTAIN") else 0.0
        elif v == e:
            s = 1.0
        elif v == "UNCERTAIN":
            s = 0.3
        else:
            s = 0.0
        (nr if e in ("CONFIRMED", "NOT_REFUTED") else rf).append(s)
    parts = [sum(c) / len(c) for c in (nr, rf) if c]
    return sum(parts) / len(parts) if parts else 0.0


async def _splice(args) -> int:
    bundle_dir = Path(args.bundle)
    data, arm_name, rows = _load_bundle(bundle_dir)
    cases = _pool_from_provenance(data)
    seed = int(data.get("seed", 0))
    trials = vb.build_trials(cases, seed=seed)
    if [(t.case_id, t.fault) for t in trials] != \
            [(r["case_id"], r["fault"]) for r in rows]:
        raise SystemExit("regenerated trial sequence does not match the "
                         "bundle — refusing to splice.")

    flag = args.select_flag
    picked = [(i, t) for i, (t, r) in enumerate(zip(trials, rows))
              if r.get(flag)]
    print(f"bundle {bundle_dir.name}: {len(rows)} trials, "
          f"{len(picked)} selected by {flag}")
    if not picked:
        print("nothing selected — nothing to measure.")
        return 0

    client = vb.EscalatingChatClient(
        args.base_url, args.main_base_url, timeout=args.timeout,
        leg=args.leg)
    verifier = Verifier(llm_client=client)
    fresh = await vb.run_trials(verifier, [t for _i, t in picked])
    await client.aclose()

    drift = 0
    spliced = [dict(r) for r in rows]
    for (i, _t), res in zip(picked, fresh):
        row = res.to_dict()
        old_cheap = rows[i].get("cheap_verdict")
        if old_cheap and row.get("cheap_verdict") \
                and row["cheap_verdict"] != old_cheap:
            drift += 1
            print(f"  ⚠ cheap-verdict drift on {row['case_id']}/"
                  f"{row['fault']}: recorded {old_cheap} → fresh "
                  f"{row['cheap_verdict']}")
        spliced[i] = row

    base_bal = _balanced(rows)
    new_bal = _balanced(spliced)
    changed = sum(1 for a, b in zip(rows, spliced)
                  if a["verdict"] != b["verdict"])
    print(f"\nrecorded config : balanced {base_bal:.3f}")
    print(f"spliced config  : balanced {new_bal:.3f}  "
          f"(Δ {new_bal - base_bal:+.3f}, {changed} verdicts changed, "
          f"{drift} cheap-verdict drifts)")
    for (i, _t), res in zip(picked, fresh):
        a, b = rows[i], res.to_dict()
        mark = "=" if a["verdict"] == b["verdict"] else "≠"
        print(f"  {mark} {a['case_id']}/{a['fault']}: {a['verdict']} → "
              f"{b['verdict']}  (expected {a['expected']})")

    out = bundle_dir.parent / (bundle_dir.name + args.suffix)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["arms"] = {arm_name: {"trials": spliced,
                                  "metrics": vb.score_trials(
                                      fresh, arm=arm_name) if False else
                                  data["arms"][arm_name]["metrics"]}}
    payload["splice"] = {
        "base_bundle": bundle_dir.name,
        "select_flag": flag,
        "n_replaced": len(picked),
        "cheap_verdict_drift": drift,
        "env": {k: os.environ.get(k, "<unset>")
                for k in ("GHOST_VERIFY_OBJECTION_CHECK",
                          "GHOST_VERIFY_OBJECTION_DISMISS",
                          "GHOST_VERIFY_TRUNCATION_GUARD",
                          "GHOST_VERIFY_OVERTURN_QUOTE",
                          "GHOST_VERIFY_TIER_ROUTING")},
        "balanced_recorded": round(base_bal, 4),
        "balanced_spliced": round(new_bal, 4),
    }
    (out / "results.json").write_text(json.dumps(payload, indent=1))
    print(f"\nwritten: {out}/results.json")
    return 0


def _rescore(args) -> int:
    bundle_dir = Path(args.bundle)
    data, _arm_name, rows = _load_bundle(bundle_dir)
    cases = _pool_from_provenance(data)
    trials = vb.build_trials(cases, seed=int(data.get("seed", 0)))
    by_key = {(t.case_id, t.fault): t for t in trials}
    have = sum(1 for r in rows if r.get("cheap_verdict"))
    if not have:
        raise SystemExit(
            "this bundle predates the cheap-verdict snapshot "
            "(2026-08-07) — rescore needs it; use splice instead.")
    changed, unknown = [], []
    for r in rows:
        cv = r.get("cheap_verdict")
        if cv != "REFUTED":
            continue                      # mechanical layer only sees refutes
        t = by_key[(r["case_id"], r["fault"])]
        sev = 0.0
        try:
            from ghost_agent.core.agent import evidence_truncation_severity
            sev = evidence_truncation_severity(t.evidence)
        except Exception:
            pass
        decision, _why, _unres = objection.resolve_refute(
            r.get("cheap_issues") or [], t.claim, t.evidence, sev)
        was_dismissed = bool(r.get("objection_dismissed"))
        was_upheld = bool(r.get("objection_upheld"))
        now_d = decision == objection.DISMISS and objection.dismiss_enabled()
        now_u = decision == objection.UPHOLD
        if (was_dismissed, was_upheld) == (now_d, now_u):
            continue
        if now_d:
            changed.append((r, "CONFIRMED (mech dismiss)"))
        elif now_u:
            changed.append((r, "REFUTED (mech uphold)"))
        elif was_dismissed or was_upheld:
            unknown.append(r)             # now escalates: outcome unrecorded
    print(f"{len(rows)} trials, {have} with cheap snapshots")
    print(f"{len(changed)} would change disposition mechanically:")
    for r, new in changed:
        print(f"  {r['case_id']}/{r['fault']}: {r['verdict']} → {new} "
              f"(expected {r['expected']})")
    print(f"{len(unknown)} would newly ESCALATE — outcome not in the "
          f"recording (top up with splice):")
    for r in unknown:
        print(f"  {r['case_id']}/{r['fault']} (expected {r['expected']})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    sp = sub.add_parser("splice")
    sp.add_argument("--bundle", required=True)
    sp.add_argument("--base-url", required=True)
    sp.add_argument("--main-base-url", required=True)
    sp.add_argument("--timeout", type=float, default=90.0)
    sp.add_argument("--leg", choices=("critic", "worker"), default="critic")
    sp.add_argument("--select-flag", default="objection_dismissed")
    sp.add_argument("--suffix", default="_spliced")
    rs = sub.add_parser("rescore")
    rs.add_argument("--bundle", required=True)
    args = ap.parse_args()
    if args.mode == "splice":
        return asyncio.run(_splice(args))
    return _rescore(args)


if __name__ == "__main__":
    raise SystemExit(main())
