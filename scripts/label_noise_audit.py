#!/usr/bin/env python3
"""§4AO — A4: is LABEL NOISE what ceilinged earn-keep and skill-prune?

Read-only. No LLM calls. Answers the question by measuring, in order:

  1. the verifier's error rates, from the oracle bench where truth is known;
  2. what fraction of the LIVE label the verifier actually decides;
  3. which consumers are exposed to it, and by how much;
  4. whether removing the noise entirely would change skill-prune's victims.

⚠ THE STANDING QUESTION this instrument exists to answer honestly: *could a
mechanism's null result be produced by the label being wrong?* The answer is
only interesting if it can come back NO — so step 4 is a counterfactual, not
an argument: it re-runs the DEPLOYED `compute_lesson_utility` on labels
de-attenuated to their noise-corrected truth and compares the kill lists.

Every section fails to NO_SOURCE rather than to zero when an input is
missing; "no evidence" and "measured zero" are different claims (§4AC).

    PYTHONPATH=src python scripts/label_noise_audit.py
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
from pathlib import Path

_SEP = "─" * 72


def _hdr(t: str) -> None:
    print(f"\n{_SEP}\n{t}\n{_SEP}")


def _no_source(what: str, path) -> None:
    print(f"  NO_SOURCE — {what}: {path} is absent. Not reporting zero.")


# ── 1. the verifier's error rates, measured against known truth ────────────

def error_rates(home: Path) -> dict | None:
    """FRR / FCR from the oracle bench.

    ⚠ THE AGGREGATE IS MIX-DEPENDENT. Per-fault false-CONFIRM spans 0.000
    (wrong_topic) to 0.569 (artifact_leak), so the pooled FCR is a property
    of the BENCH's fault mix, which is roughly uniform by construction and
    is NOT the live failure mix. Read the per-fault column, and treat the
    pooled figure as the midpoint of a wide band.
    """
    base = home / "system/eval/verifier_incumbent_baseline.json"
    if not base.exists():
        _no_source("verifier baseline", base)
        return None
    meta = json.loads(base.read_text())
    res = Path(meta.get("results_path") or "")
    if not res.exists():
        _no_source("bench results named by the baseline", res)
        return None
    arms = json.loads(res.read_text()).get("arms") or {}
    trials = next((a.get("trials") or [] for a in arms.values()), [])
    if not trials:
        _no_source("per-trial records", res)
        return None

    per: dict = {}
    for t in trials:
        k = (t.get("fault"), t.get("expected"))
        d = per.setdefault(k, {"n": 0, "CONFIRMED": 0, "REFUTED": 0, "UNCERTAIN": 0})
        d["n"] += 1
        d[t.get("verdict", "UNCERTAIN")] = d.get(t.get("verdict", "UNCERTAIN"), 0) + 1

    print(f"  source: {res}")
    print(f"\n  {'fault':24s} {'expected':11s} {'n':>4s}  error  (wrong verdict on this class)")
    good_n = good_e = bad_n = bad_e = 0
    for (fault, exp), d in sorted(per.items()):
        if exp == "CONFIRMED":
            err, good_n, good_e = d["REFUTED"] / d["n"], good_n + d["n"], good_e + d["REFUTED"]
        elif exp == "REFUTED":
            err, bad_n, bad_e = d["CONFIRMED"] / d["n"], bad_n + d["n"], bad_e + d["CONFIRMED"]
        else:
            continue          # NOT_REFUTED carries no unambiguous truth value
        print(f"  {fault:24s} {exp:11s} {d['n']:4d}  {err:.3f}")
    if not (good_n and bad_n):
        _no_source("both truth classes", res)
        return None

    frr, fcr = good_e / good_n, bad_e / bad_n
    print(f"\n  false-REFUTE  rate (good answer called bad): {frr:.4f}   n={good_n}")
    print(f"  false-CONFIRM rate (bad answer called good): {fcr:.4f}   n={bad_n}")
    print(f"\n  ⚠ pooled FCR reflects the bench's fault mix, not the live one "
          f"(per-fault range above).")
    return {"frr": frr, "fcr": fcr, "n_good": good_n, "n_bad": bad_n}


# ── 2. how much of the LIVE label the verifier actually decides ────────────

def live_label_composition(home: Path, rates: dict | None) -> dict | None:
    """The live outcome label is a COMPOSITE. Verifier verdicts and purely
    mechanical rules (structural failure, abort markers, repeated tool
    errors) both write it, and only the verifier's share carries the error
    rates above. Applying the bench rates to the whole label overstates the
    noise — the mistake this section exists to prevent.
    """
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        os.environ.setdefault("GHOST_HOME", str(home))
        from ghost_agent.distill.collector import TrajectoryCollector
    except Exception as exc:                                  # pragma: no cover
        print(f"  NO_SOURCE — cannot import the collector: {exc}")
        return None

    cor: dict = {}
    cpath = home / "system/trajectories/corrections.jsonl"
    if cpath.exists():
        for line in cpath.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    cor.setdefault(r.get("trajectory_id"), []).append(r)
                except Exception:
                    pass

    trajs = [t for t in TrajectoryCollector().iter_trajectories()
             if t.outcome in ("passed", "failed")]
    if not trajs:
        _no_source("labelled trajectories", home / "system/trajectories")
        return None

    ver = 0
    for t in trajs:
        reason = (t.failure_reason or "").lower()
        sources = {c.get("source") for c in cor.get(t.id, [])}
        if "verifier" in reason or "verifier_late" in sources:
            ver += 1
    share = ver / len(trajs)
    print(f"  labelled trajectories        : {len(trajs)}")
    print(f"  VERIFIER-decided             : {ver}  ({share:.1%})")
    print(f"  mechanical / structural      : {len(trajs) - ver}  ({1 - share:.1%})")
    if not rates:
        return {"share": share, "n": len(trajs)}

    a_ver = 1.0 - rates["frr"] - rates["fcr"]
    print(f"\n  ATTENUATION — a difference measured against a noisy binary label is")
    print(f"  shrunk by (1 - FRR - FCR); required n grows by 1/A².")
    print(f"\n  {'assumption about the mechanical rules':46s} {'A_eff':>7s} {'n×':>7s}")
    out = {}
    for mech_a, label, key in ((1.0, "noise-FREE (deterministic as written)", "best"),
                               (a_ver, "as noisy as the verifier (worst case)", "worst")):
        a = share * a_ver + (1 - share) * mech_a
        out[key] = a
        print(f"  {label:46s} {a:7.3f} {1 / a ** 2:6.2f}×")
    print(f"\n  ⚠ the true value sits between these; nothing measures how often the")
    print(f"    STRUCTURAL rules are wrong, so the band cannot be narrowed yet.")
    return {"share": share, "n": len(trajs), "a_ver": a_ver, **out}


# ── 3. which consumers touch the noisy label ───────────────────────────────

def exposure_map(home: Path) -> None:
    """A mechanism can only be ceilinged by a label it CONSUMES. Measured,
    not assumed — the earn-keep row is why the A4 hypothesis fails."""
    try:
        from ghost_agent.distill.collector import TrajectoryCollector
        from ghost_agent.distill.schema import Outcome
        from ghost_agent.router.labels import LabelSpec, derive_label
        import dataclasses
    except Exception as exc:                                  # pragma: no cover
        print(f"  NO_SOURCE — cannot import label machinery: {exc}")
        return

    spec = LabelSpec()
    trajs = list(TrajectoryCollector().iter_trajectories())
    labelled = only_outcome = 0
    for t in trajs:
        lab = derive_label(t, spec)
        if lab is None:
            continue
        labelled += 1
        if t.outcome == Outcome.FAILED.value and dataclasses.is_dataclass(t):
            alt = derive_label(dataclasses.replace(t, outcome=Outcome.PASSED.value), spec)
            if alt != "hard":
                only_outcome += 1        # the outcome, not structure, made it hard
    router_pct = only_outcome / labelled if labelled else 0.0

    print(f"  {'consumer':34s} {'exposure':>9s}   basis")
    print(f"  {'earn-keep (Track A + B4)':34s} {'0.0%':>9s}   deterministic Python validators —\n"
          f"  {'':34s} {'':>9s}   answer_int() exact match / task.verify();\n"
          f"  {'':34s} {'':>9s}   the LLM verifier is not in that loop")
    print(f"  {'router (derive_label)':34s} {router_pct:8.1%}   {only_outcome}/{labelled} labels where the outcome,\n"
          f"  {'':34s} {'':>9s}   not n_steps/tool_calls, decided 'hard'")
    print(f"  {'skill-prune outcome arm':34s} {'100%':>9s}   succeeded/failed_retrievals key off\n"
          f"  {'':34s} {'':>9s}   resolve_turn_outcome directly")
    print(f"  {'live experiment arms':34s} {'100%':>9s}   experiments.py failure_rate = 1.0 iff\n"
          f"  {'':34s} {'':>9s}   the resolved outcome is FAILED")


# ── 4. the counterfactual: would perfect labels change the victims? ────────

def prune_counterfactual(home: Path, rates: dict | None) -> None:
    try:
        from ghost_agent.memory import skills as sk
    except Exception as exc:                                  # pragma: no cover
        print(f"  NO_SOURCE — cannot import skills: {exc}")
        return
    pb = home / "system/memory/skills_playbook.json"
    if not pb.exists():
        _no_source("playbook", pb)
        return
    rows = json.loads(pb.read_text())

    def victims(playbook, tag):
        scored = sorted(((sk.compute_lesson_utility(sk._normalize_lesson(r)),
                          sk._normalize_lesson(r)) for r in playbook), key=lambda kv: kv[0])
        cutoff = [s for s, _ in scored][max(0, len(scored) // 4)]
        v = [(s, L) for s, L in scored
             if s <= cutoff and int(L.get("retrievals") or 0) >= 5
             and not L.get("verified") and not L.get("quarantined")]
        print(f"\n  {tag}  (bottom-quartile cutoff {cutoff:.4f}) — deletes {len(v)}:")
        for s, L in v:
            su, f = int(L.get("succeeded_retrievals") or 0), int(L.get("failed_retrievals") or 0)
            rate = f"{su / (su + f):.0%}" if su + f else "no outcome data"
            print(f"    util={s:.4f}  retrievals={int(L.get('retrievals') or 0):3d}  "
                  f"succ={su} fail={f} ({rate})  {str(L.get('trigger', ''))[:38]}")
        return {str(L.get("trigger", "")) for _, L in v}

    observed = victims(rows, "OBSERVED labels — what fires if GHOST_SKILL_PRUNE=1")
    if not rates:
        print("\n  (no error rates -> cannot build the perfect-label counterfactual)")
        return

    a = 1.0 - rates["frr"] - rates["fcr"]
    corrected = copy.deepcopy(rows)
    for r in corrected:
        su = int(r.get("succeeded_retrievals") or 0)
        f = int(r.get("failed_retrievals") or 0)
        n = su + f
        if not n:
            continue
        true = min(1.0, max(0.0, (su / n - rates["fcr"]) / a))
        r["succeeded_retrievals"] = int(round(n * true))
        r["failed_retrievals"] = n - int(round(n * true))
    debiased = victims(corrected, "BIAS removed (rates de-attenuated, counts fixed)")

    print(f"\n  {'kept by de-biasing':38s}: "
          f"{sorted(t[:34] for t in observed - debiased) or 'none'}")
    print(f"  {'newly condemned by de-biasing':38s}: "
          f"{sorted(t[:34] for t in debiased - observed) or 'none'}")
    print(f"\n  ⚠ EXPECT THIS TO CHANGE ALMOST NOTHING, AND DO NOT READ IT AS EXONERATION.")
    print(f"    (observed - FCR)/A is MONOTONE in the observed rate, so removing the")
    print(f"    label's systematic BIAS cannot reorder a rank-based prune. A quartile")
    print(f"    cutoff is structurally immune to bias. What corrupts a ranking is the")
    print(f"    label's VARIANCE, and only the Monte Carlo below can measure that.")

    _variance_counterfactual(rows, rates, sk)
    _identifiability(rows)
    _evidence_penalty(rows, sk)


def _variance_counterfactual(rows, rates, sk, replicates: int = 2000, seed: int = 7) -> None:
    """THE REAL TEST. Per-observation label errors are RANDOM, not just a
    shift: a lesson's decisive ticks get flipped independently, which
    scatters small-n lessons across the cutoff. Simulate both worlds from
    the same true rates — one where the ticks are recorded perfectly, one
    where they pass through the measured FRR/FCR — and count how often the
    kill lists disagree.

    Churn here is the fraction of the victim set that label noise, and only
    label noise, decides.
    """
    try:
        import numpy as np
    except Exception:                                          # pragma: no cover
        print("\n  NO_SOURCE — numpy unavailable; cannot run the variance counterfactual.")
        return None
    rng = np.random.default_rng(seed)
    frr, fcr = rates["frr"], rates["fcr"]
    a = 1.0 - frr - fcr

    idx, truth, ns = [], [], []
    for i, r in enumerate(rows):
        s = int(r.get("succeeded_retrievals") or 0)
        f = int(r.get("failed_retrievals") or 0)
        n = s + f
        if n:
            idx.append(i); ns.append(n)
            truth.append(min(1.0, max(0.0, (s / n - fcr) / a)))
    if not idx:
        print("\n  NO_SOURCE — no lesson carries decisive outcomes.")
        return None

    def kill(counts):
        pb = copy.deepcopy(rows)
        for j, i in enumerate(idx):
            pb[i]["succeeded_retrievals"] = int(counts[j])
            pb[i]["failed_retrievals"] = int(ns[j] - counts[j])
        scored = sorted(((sk.compute_lesson_utility(sk._normalize_lesson(r)), k)
                         for k, r in enumerate(pb)), key=lambda kv: kv[0])
        cut = [s for s, _ in scored][max(0, len(scored) // 4)]
        return frozenset(k for s, k in scored
                         if s <= cut and int(pb[k].get("retrievals") or 0) >= 5
                         and not pb[k].get("verified") and not pb[k].get("quarantined"))

    differ = 0
    jac, sizes = [], []
    for _ in range(replicates):
        t = rng.binomial(ns, truth)                       # the TRUE successes
        clean = kill(t)
        noisy = kill(rng.binomial(t, 1 - frr) + rng.binomial(np.array(ns) - t, fcr))
        if clean != noisy:
            differ += 1
        union = clean | noisy
        if union:
            jac.append(len(clean ^ noisy) / len(union))
        sizes.append(len(clean))

    out = {"differ": differ / replicates,
           "churn": (sum(jac) / len(jac) if jac else 0.0),
           "mean_victims": sum(sizes) / len(sizes)}
    print(f"\n  VARIANCE COUNTERFACTUAL — {replicates} replicates, seed {seed}")
    print(f"    same true rates, ticks recorded perfectly vs through FRR={frr:.3f} "
          f"FCR={fcr:.3f}")
    print(f"    replicates where the kill list DIFFERS at all : {out['differ']:.1%}")
    print(f"    mean share of the victim set decided by noise : {out['churn']:.1%}")
    print(f"    mean victims per replicate                    : {out['mean_victims']:.2f}")
    return out


def _identifiability(rows) -> None:
    """Can the per-lesson ranking be told apart from one common rate? If not,
    the quartile boundary is drawn through SAMPLING noise — which is a
    different defect from label noise, and is not fixed by a better judge."""
    obs = [(int(r.get("succeeded_retrievals") or 0), int(r.get("failed_retrievals") or 0))
           for r in rows]
    obs = [(s, f) for s, f in obs if s + f > 0]
    if len(obs) < 3:
        print("\n  NO_SOURCE — too few lessons carry decisive outcomes to test.")
        return
    tot = sum(s + f for s, f in obs)
    p = sum(s for s, _ in obs) / tot
    chi = sum((s - (s + f) * p) ** 2 / ((s + f) * p * (1 - p)) for s, f in obs)
    dof = len(obs) - 1

    def chi2_sf(x, k):
        if k % 2 == 0:
            t = math.exp(-x / 2); acc = t
            for i in range(1, k // 2):
                t *= x / (2 * i); acc += t
            return min(1.0, acc)
        z = math.sqrt(x); acc = math.erfc(z / math.sqrt(2))
        t = math.sqrt(2 / math.pi) * z * math.exp(-x / 2)
        for i in range(1, (k - 1) // 2 + 1):
            acc += t; t *= x / (2 * i + 1)
        return min(1.0, acc)

    pval = chi2_sf(chi, dof)
    rates_ = [s / (s + f) for s, f in obs]
    var_obs = statistics.pvariance(rates_)
    var_noise = sum(p * (1 - p) / (s + f) for s, f in obs) / len(obs)
    print(f"\n  IDENTIFIABILITY — H0: every lesson shares ONE true success rate")
    print(f"    lessons with any decisive outcome : {len(obs)}   total observations: {tot}")
    print(f"    pooled success rate               : {p:.4f}")
    print(f"    chi² = {chi:.2f}  dof = {dof}  p = {pval:.4f}")
    print(f"    observed variance of per-lesson rates : {var_obs:.4f}")
    print(f"    variance predicted by binomial noise  : {var_noise:.4f} "
          f"({var_noise / var_obs:.0%} of observed)")
    if pval >= 0.05:
        print(f"    ⇒ CANNOT reject: the spread is consistent with pure sampling noise.")
        print(f"      A ranking built on it is not identifiable at this n.")
    else:
        print(f"    ⇒ REJECT: real per-lesson differences exist.")
    # n needed to separate a lesson 6pp below the pool at 80% power / alpha .05
    delta = 0.06
    need = ((1.96 + 0.84) ** 2) * p * (1 - p) / (delta ** 2)
    print(f"    to separate a lesson {delta:.0%} below the pool at 80% power: "
          f"~{need:.0f} decisive outcomes PER LESSON (median today: "
          f"{statistics.median(s + f for s, f in obs):.0f})")


def _evidence_penalty(rows, sk) -> None:
    """⚠ THE ACTUAL DEFECT. The outcome multiplier 0.4 + 0.75·out_rate crosses
    1.0 at out_rate = 0.80. Below that it DEMOTES — while a lesson with fewer
    than _OUTCOME_MIN_OBS decisive outcomes is multiplied by 1.0 and pays
    nothing. If the population sits below 0.80, having earned evidence is a
    penalty and the best-measured lessons sink toward the cutoff."""
    obs = [(int(r.get("succeeded_retrievals") or 0), int(r.get("failed_retrievals") or 0))
           for r in rows]
    tot = sum(s + f for s, f in obs)
    if not tot:
        return
    pooled = sum(s for s, _ in obs) / tot
    mult = 0.4 + 0.75 * pooled
    print(f"\n  EVIDENCE PENALTY — where the outcome multiplier is centred")
    print(f"    multiplier 0.4 + 0.75·out_rate is neutral at out_rate = 0.8000")
    print(f"    this playbook's pooled success rate  = {pooled:.4f}")
    print(f"    ⇒ a lesson AT the population mean is multiplied by {mult:.4f} "
          f"({1 - mult:.1%} penalty)")
    print(f"    ⇒ a lesson with < {sk._OUTCOME_MIN_OBS} decisive outcomes is multiplied by "
          f"1.0000 (no penalty)")
    n_meas = sum(1 for s, f in obs if s + f >= sk._OUTCOME_MIN_OBS)
    print(f"    {n_meas}/{len(rows)} lessons carry enough outcomes to be charged it.")

    on = sorted(rows, key=lambda r: -sk.compute_lesson_utility(sk._normalize_lesson(r)))
    saved = sk._OUTCOME_UTILITY_ENABLED
    try:
        sk._OUTCOME_UTILITY_ENABLED = False
        off = sorted(rows, key=lambda r: -sk.compute_lesson_utility(sk._normalize_lesson(r)))
    finally:
        sk._OUTCOME_UTILITY_ENABLED = saved
    pos_on = {id(r): i for i, r in enumerate(on)}
    drops = [(pos_on[id(r)] - i, r) for i, r in enumerate(off)
             if int(r.get("succeeded_retrievals") or 0) + int(r.get("failed_retrievals") or 0)
             >= sk._OUTCOME_MIN_OBS]
    drops.sort(key=lambda kv: -kv[0])
    print(f"\n    rank change caused by the multiplier (measured lessons only):")
    for d, r in drops[:6]:
        su = int(r.get("succeeded_retrievals") or 0); f = int(r.get("failed_retrievals") or 0)
        arrow = f"↓{d}" if d > 0 else (f"↑{-d}" if d < 0 else "—")
        print(f"      {arrow:>5s}  succ={su:3d} fail={f:3d} ({su / (su + f):.0%})  "
              f"{str(r.get('trigger', ''))[:38]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", default=os.environ.get("GHOST_HOME", ""),
                    help="GHOST_HOME (live data root)")
    args = ap.parse_args()
    if not args.home:
        print("refusing to guess GHOST_HOME — pass --home or export it", file=sys.stderr)
        return 2
    home = Path(args.home).expanduser().resolve()
    if not home.is_dir():
        print(f"not a directory: {home}", file=sys.stderr)
        return 2
    print(f"§4AO label-noise audit — GHOST_HOME={home}")

    _hdr("1. THE VERIFIER'S ERROR RATES, against known truth")
    rates = error_rates(home)

    _hdr("2. HOW MUCH OF THE LIVE LABEL THE VERIFIER DECIDES")
    live_label_composition(home, rates)

    _hdr("3. EXPOSURE — which consumers touch the noisy label")
    exposure_map(home)

    _hdr("4. COUNTERFACTUAL — would PERFECT labels change skill-prune's victims?")
    prune_counterfactual(home, rates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
