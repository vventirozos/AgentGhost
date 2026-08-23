#!/usr/bin/env python3
"""§4CM D4b — what would the replay engine's verdicts be worth ON THE LIVE CORPUS?

THE QUESTION D4 COULD NOT ANSWER. The seeded gate certifies that the
paired-verdict MACHINERY is operative: fork, setup, tool restriction,
agent run, validator, decision. It cannot certify that a `mattered_*`
verdict means anything, for two measured reasons.

  * **Wrong kind.** Its seeded positive is `tool_ablate`, and ZERO of the
    309 specs the engine actually plans are that kind (131
    `lesson_withhold`, 150 `verify_toggle`, 28 `step_deny`). Its effect
    size is 1.0 by construction; theirs are small.
  * **Wrong REGIME, which is the sharper one.** The paired rule's
    false-`mattered` rate is `2·pⁿ·qⁿ / (pⁿ+qⁿ)²` for a task with
    per-run pass probability `p`. The seed corpus is `stable-pass`
    self-play challenges, so 15 of 17 measured cases ran at `p = 1.0`
    — where that rate is **0.000 by construction**. A specificity of
    1.00 measured there is close to an arithmetic identity. The live
    corpus is not known to be in that regime.

    ⚠ AND A CORRECTION TO THE ARGUMENT THAT MOTIVATED THIS SCRIPT. I
    reached for "the live corpus's control legs agreed with the recording
    72.7% of the time, so p ≈ 0.727 and the floor is 0.096" — and those
    are two different quantities. Agreement-with-the-RECORDING can be
    low while leg-to-leg behaviour is perfectly deterministic: an episode
    that fails identically on all six legs disagrees with a `passed`
    recording every time and has p = 0, where the floor is ZERO. The
    per-run pass rate has never been measured on this corpus. That is
    what this script is for, and quoting 0.727 as if it were that number
    was the same species of error as the regime it was written to expose.

So this script measures the regime instead of assuming it. It runs `k`
IDENTICAL control legs on real episodes — no perturbation, no second
condition, nothing to apply — and reads two numbers off them:

  1. **p̂, the per-episode pass rate.** From it, in closed form: the
     probability the paired rule DECIDES at all, `(pⁿ+qⁿ)²`, and the
     probability it calls noise an effect, `2pⁿqⁿ/(pⁿ+qⁿ)²`.
  2. **The A/A false-positive rate, empirically, from the same legs.**
     Split the `k` legs every possible way into two arms of `n` and run
     the REAL `decide_verdict` on each split. Every arm is the same
     condition, so any `mattered_*` is a false positive by construction.
     With k=6 and n=3 that is exhaustive: 20 splits, and a false
     `mattered` is possible ONLY when exactly 3 of the 6 legs passed,
     in 2 of the 20 splits.

The comparison between (1) and (2) is the point. If the closed form
predicts what the splits produce, then `noise_floor` — already recorded
on every credit row — is a VALIDATED per-row trust score, and
`dream_credit` can open conditionally on it instead of on a blanket
pass. That is a more useful gate than the one it replaces.

⚠ WHY THE TWO NUMBERS ARE COMPARABLE AT ALL, since they condition
differently. The closed form averages over the randomness of the legs;
the split rate conditions on the legs actually observed. Per episode they
differ — an episode that came out 3-of-6 has a split rate of 0.100 while
its p̂ = 0.5 predicts 0.031. But in EXPECTATION they are identical, and
exactly so: with 6 legs a false `mattered` requires exactly 3 passes,
which happens with probability `C(6,3)·p³q³ = 20p³q³`, and then occurs in
1 of the 10 unordered splits — so `E[split rate] = 20p³q³ · 0.1 = 2p³q³`,
which is the closed form. That identity is what makes the corpus-level
`formula_error` a real check rather than a comparison of two different
things, and it is why the aggregate is the only level at which the two
may be compared.

⚠ THE SPLITS ARE NOT INDEPENDENT SAMPLES. They share legs. The estimator
aggregates PER EPISODE (the mean over that episode's splits is one
observation of that episode's false-positive probability) and then over
episodes. Treating 20 splits as 20 samples would understate the error by
about the square root of the overlap.

    GHOST_HOME=... PYTHONPATH=src python scripts/replay_noise_floor.py \\
        --episodes 25 --legs 6

Exit: 0 = measured; 2 = no data / not runnable; 4 = incomplete.
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

#: Legs per episode. 6 with arms of 3 is the cheapest count that both
#: estimates a pass rate and admits an EXHAUSTIVE A/A split enumeration.
DEFAULT_LEGS = 6
#: Episodes below which nothing is reported as a corpus statistic.
MIN_EPISODES = 12

EXIT_OK = 0
EXIT_NO_DATA = 2
EXIT_INCOMPLETE = 4


def analytic(p: float, n: int) -> dict:
    """Closed form for the paired rule at pass rate `p`, `n` legs an arm."""
    q = 1.0 - p
    pn, qn = p ** n, q ** n
    decided = (pn + qn) ** 2
    mattered = 2.0 * pn * qn
    return {
        "p_decided": round(decided, 4),
        "p_false_mattered": round(mattered, 4),
        "p_false_mattered_given_decided": (round(mattered / decided, 4)
                                           if decided > 0 else None),
    }


def aa_splits(outcomes: list, n: int) -> dict:
    """Every way to deal `2n` identical legs into two arms of `n`, judged
    by the REAL decision rule.

    Returns counts, not a rate: the caller aggregates per episode,
    because splits of one episode share legs and are not independent.
    """
    from ghost_agent.core.replay_engine import (
        ReplayLeg, decide_verdict, VERDICT_MATTERED_POS,
        VERDICT_MATTERED_NEG, VERDICT_NO_EFFECT, VERDICT_ABSTAIN,
    )
    counts = {VERDICT_MATTERED_POS: 0, VERDICT_MATTERED_NEG: 0,
              VERDICT_NO_EFFECT: 0, VERDICT_ABSTAIN: 0}
    idx = list(range(len(outcomes)))
    seen = set()
    for combo in itertools.combinations(idx, n):
        rest = tuple(i for i in idx if i not in combo)
        key = tuple(sorted((combo, rest)))
        if key in seen:                  # arms are unordered
            continue
        seen.add(key)
        a = [ReplayLeg(arm="control", passed=outcomes[i], applied=True)
             for i in combo]
        b = [ReplayLeg(arm="perturbed", passed=outcomes[i], applied=True)
             for i in rest]
        verdict, _why = decide_verdict(a, b)
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts


async def _validator_only_in(context, validator: str, source_workspace,
                             leg_timeout_s: float):
    """The negative control, run in THE REGIME THE LEGS RUN IN.

    ⚠ `RE.run_validator_only` hardcodes an EMPTY fork. That was a matched
    screen while the legs were empty too; with `--fork live` it is not.
    A seeded fork is a COPY OF THE LIVE SANDBOX, so a validator asserting
    an artifact that is still on disk FAILS the empty screen — reading as
    "this check discriminates" — and then PASSES every seeded leg without
    the agent doing anything. That episode scores p̂ = 1.0, lands in
    `always_pass`, contributes a guaranteed zero to the floor, and
    measured nothing at all.

    The engine has the same asymmetry (`run_batch` screens empty at
    :2035 and runs legs live-by-default at :2062), so this does not just
    fix the census — it MEASURES the engine's blind spot, which is why
    both screens are run and both are recorded.
    """
    import asyncio
    from ghost_agent.core import replay_engine as RE
    from ghost_agent.core.isolation import isolated_replay_context

    if source_workspace == RE.USE_LIVE_WORKSPACE:
        source_workspace = getattr(context, "sandbox_dir", None)
    leg = RE.ReplayLeg(arm="negative_control_seeded")
    try:
        async with isolated_replay_context(
                context, network="none", label="dream-negctl-seeded",
                source_workspace=source_workspace) as run:
            vpath = run.workspace / RE.VALIDATOR_FILENAME
            await asyncio.to_thread(vpath.write_text, validator)
            out, code = await asyncio.to_thread(
                run.sandbox.execute,
                f"python3 {RE.VALIDATOR_FILENAME}", RE.VALIDATOR_TIMEOUT_S)
            leg.validator_exit = int(code)
            leg.validator_output = str(out or "")[:1000]
            if "[SANDBOX INFRA ERROR" in leg.validator_output:
                leg.reason = "sandbox infra fault during the seeded control"
            elif code in (0, 1):
                leg.passed = (code == 0)
    except Exception as exc:                # noqa: BLE001
        leg.reason = f"{type(exc).__name__}: {exc}"
    return leg


async def measure_episode(context, traj, tri, *, legs: int,
                          leg_timeout_s: float,
                          source_workspace=None) -> dict:
    """`legs` identical control runs of one real episode.

    ``source_workspace`` is THE REGIME, and it is not a detail. The
    engine's own `run_spec` defaults it to `USE_LIVE_WORKSPACE`, so every
    leg it runs starts from a copy of the live sandbox; this census used
    to hardcode `None`, so its legs started EMPTY. A floor measured in a
    world the rule never runs in is a number about the wrong regime.
    Passed through, recorded on the run, never assumed.
    """
    from ghost_agent.core import replay_engine as RE

    row = {"trajectory_id": str(getattr(traj, "id", "") or ""),
           "recorded_outcome": str(getattr(tri, "recorded_outcome", "")
                                   or getattr(traj, "outcome", "") or "")}
    llm = getattr(context, "llm_client", None)
    # ⚠ TIME EVERY PHASE, INCLUDING THE ONES THAT END IN A SKIP. Without
    # this the run's cost is one undifferentiated number and the only way
    # to size the NEXT run is to guess how it split — which is how the
    # first projection had to be made. A skipped episode still pays for
    # synthesis and often for a negative control; those seconds are the
    # difference between a three-hour estimate and a nine-hour one.
    _t0 = time.monotonic()
    _synth = {}
    validator = await RE.synthesize_validator(traj, llm, out=_synth)
    row["synth_seconds"] = round(time.monotonic() - _t0, 1)
    if not validator:
        row["skip"] = "no_admissible_validator"
        # ⚠ WHY, not just THAT. This is the funnel's largest loss (46 of
        # 128) and a bare count cannot tell a model that returns nothing
        # from a screen that rejects what it returns — opposite fixes.
        row["synth_reject"] = _synth.get("reason") or "unknown"
        row["synth_raw_chars"] = _synth.get("raw_chars", 0)
        return row
    # The SAME negative control the engine uses: a validator that passes
    # an empty fork agrees with every `passed` episode for free.
    _t0 = time.monotonic()
    neg = await RE.run_validator_only(context, validator,
                                      leg_timeout_s=leg_timeout_s)
    row["neg_control_seconds"] = round(time.monotonic() - _t0, 1)
    row["neg_empty_exit"] = neg.validator_exit
    row["neg_empty_passed"] = neg.passed
    if neg.validator_exit == 2:
        row["skip"] = "episode_not_filesystem_checkable"
        return row
    if neg.passed is None:
        row["skip"] = "negative_control_inconclusive"
        return row
    if neg.passed:
        row["skip"] = "validator_did_not_discriminate"
        return row
    # …and the same question asked in the world the LEGS will run in.
    if source_workspace is not None:
        _t0 = time.monotonic()
        seeded = await _validator_only_in(context, validator,
                                          source_workspace, leg_timeout_s)
        row["neg_seeded_seconds"] = round(time.monotonic() - _t0, 1)
        row["neg_seeded_exit"] = seeded.validator_exit
        row["neg_seeded_passed"] = seeded.passed
        if seeded.passed is None:
            row["skip"] = "negative_control_inconclusive"
            return row
        if seeded.passed:
            # ⚠ THE ENGINE WOULD HAVE ADMITTED THIS. Its screen is the
            # empty one, which this episode already passed. Counting it
            # is the point: it sizes the engine's blind spot.
            row["engine_screen_would_admit"] = True
            row["skip"] = "vacuous_in_the_regime_the_legs_run_in"
            return row

    spec = {"spec_id": f"aa-{row['trajectory_id']}",
            "trajectory_id": row["trajectory_id"],
            "perturbation": "", "target": "",
            "fork_step": int(getattr(tri, "fork_step", 0) or 0),
            "user_request": str(getattr(traj, "user_request", "") or ""),
            "recorded_outcome": row["recorded_outcome"],
            "n_steps": int(getattr(tri, "n_steps", 0) or 0)}

    row["validator_sha"] = __import__("hashlib").sha1(
        validator.encode("utf-8", "replace")).hexdigest()[:12]
    outcomes, durations = [], []
    for _ in range(legs):
        # ⚠ arm="control" EVERY TIME. No perturbation is constructed, so
        # nothing can be half-applied and `applied` cannot be corrupted —
        # which is why this needs no engine change at all.
        leg = await RE.run_leg(context, spec, arm="control",
                               validator=validator,
                               source_workspace=source_workspace,
                               leg_timeout_s=leg_timeout_s)
        outcomes.append(leg.passed)
        durations.append(round(getattr(leg, "duration_s", 0.0) or 0.0, 1))
        if leg.passed is None:
            # ⚠ THE OUTCOME IS ALREADY DECIDED. An A/A episode needs ALL
            # `legs` graded — one ungradable leg skips the episode below,
            # whatever the remaining legs do — so every leg after this
            # one is provably wasted work. MEASURED: one episode spent
            # 1,052 s (6 legs at the 240 s cap) to reach a skip that leg
            # one had already determined, 81% of that run's whole cost.
            row["aborted_after"] = len(outcomes)
            break
    row["legs"] = outcomes
    row["leg_seconds"] = durations
    row["legs_run"] = len(outcomes)
    gradable = [o for o in outcomes if o is not None]
    row["ungradable"] = len(outcomes) - len(gradable)
    if len(gradable) < legs:
        # A/A needs all `legs` graded: an ungradable leg is an abstain in
        # every split it lands in, which would masquerade as a low
        # false-positive rate.
        row["skip"] = "ungradable_leg"
        return row
    p_hat = sum(1 for o in gradable if o) / len(gradable)
    row["pass_rate"] = round(p_hat, 4)
    row["analytic"] = analytic(p_hat, legs // 2)
    row["splits"] = aa_splits(gradable, legs // 2)
    return row


async def measure(context, *, episodes: int = 25, legs: int = DEFAULT_LEGS,
                  leg_timeout_s: float = 240.0, run_deadline: float = None,
                  rows_path=None, seeded_forks: bool = True,
                  only: set = None) -> dict:
    """Run the A/A census over real episodes."""
    from ghost_agent.core import replay_engine as RE

    out = {"rows": [], "skipped": {}, "episodes_seen": 0,
           "legs_per_episode": legs, "stopped_early": "",
           "arms": legs // 2, "ships_n_pairs": RE.DEFAULT_N_PAIRS,
           "fork_regime": "live" if seeded_forks else "empty",
           # ⚠ A TARGETED SUBSET IS NOT A CORPUS MEASUREMENT, and this
           # file accumulates across runs by design — so the run says so
           # in its own header rather than leaving a later reader to
           # infer it from the row count.
           "targeted_subset": sorted(only) if only else []}
    src_ws = RE.USE_LIVE_WORKSPACE if seeded_forks else None
    # ⚠ WHICH HALF OF THE CORPUS EACH EPISODE CAME FROM. `EpisodeSource`
    # yields real episodes first and then bench, with no marker on the
    # trajectory — and the two are 67 and 84 of the 151 replayable ones.
    # They are different populations: a bench episode carries its OWN
    # executable validator and is a bank solve, a real one gets a
    # synthesised check. Blending them into one floor would be the D4
    # mistake again — a number measured in one regime and reported as
    # the corpus's. Cheap to separate: triage is a pure read.
    real_ids = set()
    try:
        for _t, _tri in RE.EpisodeSource(
                args=getattr(context, "args", None),
                include_bench=False).iter_episodes():
            real_ids.add(str(getattr(_t, "id", "") or ""))
    except Exception:                       # noqa: BLE001
        real_ids = set()
    out["corpus_real"] = len(real_ids)
    src = RE.EpisodeSource(args=getattr(context, "args", None))
    run_id = f"{os.getpid()}-{int(time.time())}"
    if rows_path:
        try:
            with open(rows_path, "a") as fh:
                fh.write(json.dumps({"__run__": "start", "run_id": run_id,
                                     "legs": legs,
                                     "fork_regime": out["fork_regime"],
                                     "targeted_subset":
                                         out["targeted_subset"],
                                     "leg_timeout_s": leg_timeout_s,
                                     "episodes": episodes}) + "\n")
        except Exception:
            pass

    interrupted = ""
    try:
        for traj, tri in src.iter_episodes():
            if only is not None:
                tid = str(getattr(traj, "id", "") or "")
                if not any(tid.startswith(x) for x in only):
                    continue
            if len(out["rows"]) >= episodes:
                break
            if run_deadline is not None and time.monotonic() >= run_deadline:
                out["stopped_early"] = "run deadline reached"
                break
            ok, why = RE.preflight()
            if not ok:
                out["stopped_early"] = f"preflight stood down mid-run: {why}"
                print(f"  STOPPING: {why}", flush=True)
                break
            out["episodes_seen"] += 1
            try:
                row = await measure_episode(context, traj, tri, legs=legs,
                                            leg_timeout_s=leg_timeout_s,
                                            source_workspace=src_ws)
                row["source"] = ("real"
                                 if str(getattr(traj, "id", "") or "") in real_ids
                                 else "bench")
            except Exception as exc:            # noqa: BLE001
                row = {"trajectory_id": str(getattr(traj, "id", "") or ""),
                       "skip": f"raised: {type(exc).__name__}"}
            row["run_id"] = run_id
            if row.get("skip"):
                out["skipped"][row["skip"]] = out["skipped"].get(row["skip"],
                                                                 0) + 1
                print(f"  {row['trajectory_id'][:16]:16} SKIP {row['skip']}",
                      flush=True)
            else:
                out["rows"].append(row)
                a = row["analytic"]
                sp = row["splits"]
                fm = sp.get("mattered_pos", 0) + sp.get("mattered_neg", 0)
                print(f"  {row['trajectory_id'][:16]:16} legs={row['legs']} "
                      f"p={row['pass_rate']:.2f} "
                      f"predicted_fp={a['p_false_mattered']:.3f} "
                      f"observed_fp={fm}/{sum(sp.values())}", flush=True)
            if rows_path:
                try:
                    with open(rows_path, "a") as fh:
                        fh.write(json.dumps(row) + "\n")
                except Exception:
                    pass
    except BaseException as exc:            # noqa: BLE001
        # ⚠ THE FOOTER MUST SURVIVE AN INTERRUPT. `rescore` reads its
        # absence as "the run did not finish", which is right — but a
        # ten-hour run killed at hour nine has nine hours of usable rows,
        # and without a footer they score as a run that never ended. The
        # footer is written on EVERY exit and records HOW it ended, so it
        # can never claim a completion that did not happen.
        interrupted = f"{type(exc).__name__} mid-run"
        out["stopped_early"] = out["stopped_early"] or interrupted
        raise
    finally:
        if rows_path:
            try:
                with open(rows_path, "a") as fh:
                    fh.write(json.dumps({
                        "__run__": "end", "run_id": run_id,
                        "stopped_early": out["stopped_early"],
                        "interrupted": interrupted,
                        "episodes_seen": out["episodes_seen"]}) + "\n")
            except BaseException:           # noqa: BLE001
                # Not `Exception`: a KeyboardInterrupt landing here is
                # exactly the case that loses the footer.
                pass
    return out


def rescore(rows_path) -> dict:
    """Rebuild the scoreboard from `aa_rows.jsonl`.

    ⚠ FILTERED BY `run_id`, because the file ACCUMULATES. This scratch
    path is fixed and the writer appends, so it already holds rows from
    three runs — and D4 learned this the expensive way: its recovery
    command merged two runs into one scoreboard and reported PASS at
    exit 0. Only the last run's rows are scored, and a file with no
    usable run id is scoreable for inspection and never for a verdict.
    """
    raw = []
    for line in Path(rows_path).read_text().splitlines():
        if line.strip():
            try:
                raw.append(json.loads(line))
            except Exception:
                continue
    markers = [r for r in raw if r.get("__run__")]
    ids = [r.get("run_id") for r in markers if r.get("run_id")]
    last = ids[-1] if ids else None
    rows = [r for r in raw if not r.get("__run__")
            and (last is None or r.get("run_id") == last)]
    dropped = len([r for r in raw if not r.get("__run__")]) - len(rows)
    header = next((r for r in reversed(markers)
                   if r.get("__run__") == "start"
                   and (last is None or r.get("run_id") == last)), None)
    footer = next((r for r in reversed(markers)
                   if r.get("__run__") == "end"
                   and last is not None and r.get("run_id") == last), None)
    skipped = {}
    for r in rows:
        if r.get("skip"):
            skipped[r["skip"]] = skipped.get(r["skip"], 0) + 1
    out = {
        "rows": [r for r in rows if not r.get("skip") and r.get("splits")],
        "skipped": skipped, "episodes_seen": len(rows),
        "legs_per_episode": (header or {}).get("legs"),
        "arms": ((header or {}).get("legs") or 0) // 2,
        "stopped_early": "", "rescored_from": str(rows_path),
        "rescored_run_id": last, "rescored_dropped_rows": dropped,
        "fork_regime": (header or {}).get("fork_regime") or "unrecorded",
    }
    if last is None:
        out["stopped_early"] = ("the rows file has no usable run id — it "
                                "pre-dates the markers, or holds rows from "
                                "runs that cannot be told apart")
    elif footer is None:
        out["stopped_early"] = ("no completion footer — the run did not "
                                "finish (killed, crashed, or interrupted)")
    else:
        out["stopped_early"] = str(footer.get("stopped_early") or "")
    return out


def score(out: dict) -> dict:
    """Corpus-level numbers, aggregated PER EPISODE."""
    rows = [r for r in out.get("rows") or [] if r.get("splits")]
    res = {"n_episodes": len(rows), "legs_per_episode":
           out.get("legs_per_episode")}
    # ⚠ THE CENSUS SURVIVES A RUN THAT MEASURED NOTHING. It used to sit
    # below the NO DATA return, so a corpus that produced zero usable
    # episodes lost the one thing it DID produce — the reason. "The
    # corpus is thin" and "the filter is too tight" give the same
    # episode count and need opposite responses.
    res["skipped"] = dict(out.get("skipped") or {})
    # ⚠ THE SCOREBOARD MUST NAME THE WORLD IT MEASURED. Two rescored
    # files, one `live` and one `empty`, were textually indistinguishable
    # — in a script whose whole thesis is that a floor from the wrong
    # regime is a number about the wrong regime.
    res["fork_regime"] = out.get("fork_regime") or "unrecorded"
    res["targeted_subset"] = list(out.get("targeted_subset") or [])
    res["excluded_for_flakiness"] = int(
        (out.get("skipped") or {}).get("ungradable_leg", 0))
    if not rows:
        res["verdict"] = ("INCOMPLETE" if out.get("stopped_early")
                          else "NO DATA")
        return res
    rates = [r["pass_rate"] for r in rows]
    res["pass_rate_median"] = sorted(rates)[len(rates) // 2]
    res["pass_rate_deterministic"] = sum(1 for p in rates
                                         if p >= 0.999 or p <= 0.001)
    # ⚠ p = 0 AND p = 1 ARE NOT THE SAME KIND OF DETERMINISM, and the
    # difference decides whether an episode is worth replaying at all.
    # At p = 1 a perturbation CAN show an effect: control passes,
    # perturbed fails. At p = 0 the control arm never passes, so the
    # engine abstains on "nothing to break" for every perturbation of
    # that episode, forever. Those episodes are dead weight in the
    # corpus and nothing has ever counted them.
    res["always_pass"] = sum(1 for p in rates if p >= 0.999)
    res["never_pass"] = sum(1 for p in rates if p <= 0.001)
    res["stochastic"] = len(rates) - res["always_pass"] - res["never_pass"]
    res["by_source"] = {}
    for src in ("real", "bench"):
        sub = [r for r in rows if r.get("source") == src]
        if not sub:
            continue
        sr = [r["pass_rate"] for r in sub]
        s_fp = []
        for r in sub:
            sp = r["splits"]
            tot = sum(sp.values()) or 1
            s_fp.append((sp.get("mattered_pos", 0)
                         + sp.get("mattered_neg", 0)) / tot)
        res["by_source"][src] = {
            "n": len(sub),
            "always_pass": sum(1 for p in sr if p >= 0.999),
            "never_pass": sum(1 for p in sr if p <= 0.001),
            "stochastic": sum(1 for p in sr if 0.001 < p < 0.999),
            "false_mattered_rate": round(sum(s_fp) / len(s_fp), 4),
        }
    # ⚠ THE FLOOR THE ENGINE ACTUALLY EXPERIENCES IS NOT THIS CORPUS'S.
    # The census admits an episode when its validator DISCRIMINATES (it
    # fails an empty fork). The engine admits one only when the control
    # arm also REPRODUCES the recorded outcome — its agreement test. So
    # every episode here whose legs never reproduce is one the engine
    # would refuse, and its guaranteed zero false-mattereds dilutes the
    # headline with a population the rule never runs on. Measured on run
    # 3: 17 of 23 episodes never reproduced, so the undiluted number was
    # dominated by episodes the engine cannot use.
    #
    # ⚠ `p_admit` IS A SINGLE LEG, NOT AN ARM, and the first version got
    # this wrong in the flattering direction. `run_batch` admits an
    # episode on ONE control leg (replay_engine.py `control = await
    # run_leg(...)`, then `if control.passed != (recorded == "passed")`).
    # `run_spec`'s n control legs are only required to be GRADABLE, never
    # unanimous — the unanimity requirement lives in `decide_verdict`,
    # and the split enumeration in `fp` already contains it. Weighting by
    # p̂³ therefore modelled a gate the engine does not have AND
    # double-counted one it does.
    #
    # The error was not random. With 6 legs, fp > 0 iff exactly 3 legs
    # passed, i.e. iff p̂ = 0.5 — and w³ = w exactly at w ∈ {0, 1}. So
    # cubing left every zero-fp episode at full weight and cut the weight
    # of the ONLY episodes that produce the quantity being measured by
    # 4×, biasing the headline low.
    n_arm = 1
    adm, adm_w, adm_fp = [], 0.0, 0.0
    for r in rows:
        p_hat = r["pass_rate"]
        want_pass = str(r.get("recorded_outcome", "")).lower() == "passed"
        p_admit = (p_hat ** n_arm) if want_pass else ((1 - p_hat) ** n_arm)
        r["p_admit"] = round(p_admit, 6)
        sp = r["splits"]
        tot = sum(sp.values()) or 1
        fp = (sp.get("mattered_pos", 0) + sp.get("mattered_neg", 0)) / tot
        if p_admit > 0:
            adm.append(r)
        adm_w += p_admit
        adm_fp += p_admit * fp
    res["engine_admissible"] = {
        "n": len(adm),
        "n_refused_by_agreement": len(rows) - len(adm),
        "admission_legs": n_arm,
        # Unweighted over episodes the engine could ever admit…
        "false_mattered_rate": (round(sum(
            (r["splits"].get("mattered_pos", 0)
             + r["splits"].get("mattered_neg", 0))
            / (sum(r["splits"].values()) or 1) for r in adm) / len(adm), 4)
            if adm else None),
        # …and weighted by HOW OFTEN each would be admitted, which is the
        # rate a consumer of the ledger actually meets.
        "admission_weighted": (round(adm_fp / adm_w, 4)
                               if adm_w > 0 else None),
        "total_admission_weight": round(adm_w, 3),
    }
    # ⚠ PER EPISODE, then over episodes. The splits of one episode share
    # legs; averaging all splits pooled would understate the error.
    pred_dec, pred_fp, obs_dec, obs_fp = [], [], [], []
    for r in rows:
        a, sp = r["analytic"], r["splits"]
        total = sum(sp.values()) or 1
        decided = total - sp.get("abstain", 0)
        fm = sp.get("mattered_pos", 0) + sp.get("mattered_neg", 0)
        pred_dec.append(a["p_decided"])
        pred_fp.append(a["p_false_mattered"])
        obs_dec.append(decided / total)
        obs_fp.append(fm / total)
    def _m(xs):
        return round(sum(xs) / len(xs), 4) if xs else None

    def _se(xs):
        """Standard error over EPISODES, which is the unit of
        independence: an episode's splits share legs.

        ⚠ TWO HONEST QUALIFICATIONS, because the obvious claim for this
        is wrong. (1) With a fixed leg count every episode contributes
        the same number of splits, so the pooled MEAN and the per-episode
        mean are arithmetically IDENTICAL — the framing does not protect
        the point estimate. (2) An earlier version of this comment said
        pooling would understate the error "by about sqrt(10)". Measured,
        it does not: the design effect is `1 + (m-1)·ICC`, and the splits
        within an episode are HETEROGENEOUS (a 3-of-6 episode has one
        false `mattered` and nine abstains), so the intra-cluster
        correlation is low and the two standard errors come out close.
        The episode is still the right denominator in principle — this is
        the number to report — but the difference is small, and claiming
        a factor nobody measured is the kind of thing this file exists to
        stop doing.
        """
        if len(xs) < 2:
            return None
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return round(math.sqrt(var / len(xs)), 4)
    res["predicted_decided_rate"] = _m(pred_dec)
    res["observed_decided_rate"] = _m(obs_dec)
    res["predicted_false_mattered_rate"] = _m(pred_fp)
    res["observed_false_mattered_rate"] = _m(obs_fp)
    res["observed_false_mattered_se"] = _se(obs_fp)
    res["n_independent_units"] = len(obs_fp)
    # The headline: does the closed form describe what the splits did?
    if res["predicted_false_mattered_rate"] is not None:
        res["formula_error"] = round(
            res["observed_false_mattered_rate"]
            - res["predicted_false_mattered_rate"], 4)
    dec = res["observed_decided_rate"] or 0.0
    res["false_mattered_given_decided"] = (
        round((res["observed_false_mattered_rate"] or 0.0) / dec, 4)
        if dec > 0 else None)
    # ⚠ THE SELECTION BIAS, NAMED. An episode is skipped if ANY of its
    # legs is ungradable — an infra fault, a timeout, an inconclusive
    # validator. Those are exactly the flaky episodes, so the measured
    # population is the STABLE tail and the floor reported here is a
    # LOWER bound on the corpus's. Counted above so the direction is
    # visible even when nothing was measurable at all.
    res["powered"] = len(rows) >= MIN_EPISODES
    if out.get("stopped_early"):
        res["verdict"] = "INCOMPLETE"
        return res
    res["verdict"] = ("MEASURED" if res["powered"] else "UNDERPOWERED")
    return res


def render(out: dict, res: dict) -> None:
    print("§4CM D4b — the replay engine's noise floor ON THE LIVE CORPUS")
    print(f"  fork regime: {res.get('fork_regime')}"
          + ("   (legs start from a COPY of the live sandbox, which is "
             "what the engine's run_spec does)"
             if res.get("fork_regime") == "live" else
             "   ⚠ NOT the regime the engine ships"
             if res.get("fork_regime") == "empty" else
             "   ⚠ the rows do not say which world this was measured in"))
    if res.get("targeted_subset"):
        print(f"  ⚠ TARGETED SUBSET of "
              f"{len(res['targeted_subset'])} episode(s) — NOT a corpus "
              f"measurement. The floor below describes these episodes "
              f"only, and they were chosen because they FAILED before, "
              f"so they are not a random sample of anything.")
    print(f"  episodes measured: {res['n_episodes']} of "
          f"{out.get('episodes_seen', 0)} seen   "
          f"legs per episode: {res.get('legs_per_episode')}")
    if out.get("skipped"):
        print("  skipped: " + ", ".join(
            f"{n}x {k}" for k, n in sorted(out["skipped"].items(),
                                           key=lambda kv: -kv[1])))
    if out.get("rescored_from"):
        print(f"  (rescored from {out['rescored_from']}, run "
              f"{out.get('rescored_run_id') or 'unknown'}"
              + (f"; {out['rescored_dropped_rows']} row(s) from EARLIER "
                 f"runs in the same file were ignored"
                 if out.get("rescored_dropped_rows") else "") + ")")
    if out.get("stopped_early"):
        print(f"  ⚠ STOPPED EARLY: {out['stopped_early']}")
    if not res.get("n_episodes"):
        print("  VERDICT: NO DATA")
        return
    print(f"  per-episode pass rate: median {res['pass_rate_median']:.2f}"
          f"   always-pass {res['always_pass']}  never-pass "
          f"{res['never_pass']}  stochastic {res['stochastic']}")
    if res["never_pass"]:
        print(f"     ⚠ the {res['never_pass']} never-pass episode(s) are "
              f"DEAD WEIGHT: the control arm never passes, so the engine "
              f"abstains on 'nothing to break' for every perturbation of "
              f"them, forever")
    for src, d in (res.get("by_source") or {}).items():
        print(f"     {src:5}: n={d['n']:3} always-pass {d['always_pass']:3} "
              f"never-pass {d['never_pass']:3} stochastic "
              f"{d['stochastic']:3}  false-mattered "
              f"{d['false_mattered_rate']:.3f}")
    if len(res.get("by_source") or {}) < 2:
        # ⚠ MEASURED, not a constant. Two different hardcoded splits had
        # already drifted into this file, and one of them was printed to
        # the operator as fact.
        cr = out.get("corpus_real")
        known = (f"({cr} real episodes were counted before the run)"
                 if cr else "⚠ the real-corpus probe returned NOTHING, so "
                            "every row may be labelled `bench` wrongly")
        print("     ⚠ ONE HALF OF THE CORPUS ONLY. `EpisodeSource` yields "
              f"real episodes before bench ones {known}, so a short run "
              "measures the real half and says nothing about the other.")
    ea = res.get("engine_admissible") or {}
    if ea.get("n") is not None:
        print(f"  ENGINE-ADMISSIBLE ({ea['admission_legs']} control "
              f"leg matching the recording — the engine's own gate):")
        print(f"     {ea['n']} of {res['n_episodes']} episodes could ever "
              f"be admitted; {ea['n_refused_by_agreement']} would be "
              f"refused by the agreement test")
        if ea.get("false_mattered_rate") is not None:
            print(f"     false-mattered over those episodes: "
                  f"{ea['false_mattered_rate']:.4f}")
        if ea.get("admission_weighted") is not None:
            print(f"     weighted by how often each IS admitted: "
                  f"{ea['admission_weighted']:.4f}   "
                  f"(total weight {ea['total_admission_weight']:.2f} "
                  f"episode-equivalents)")
        print("     ⚠ THIS, not the corpus-wide number, is the rate a "
              "consumer of the credit ledger meets. An episode that never "
              "reproduces contributes a guaranteed zero to the corpus "
              "figure and is one the engine refuses to run.")
    if res.get("excluded_for_flakiness"):
        print(f"     ⚠ {res['excluded_for_flakiness']} episode(s) excluded "
              f"for an ungradable leg — those are the FLAKY ones, so the "
              f"floor below is a LOWER bound on the corpus's")
    print(f"  DECIDES at all      : predicted "
          f"{res['predicted_decided_rate']:.3f}   observed "
          f"{res['observed_decided_rate']:.3f}")
    _se_txt = ("" if res.get("observed_false_mattered_se") is None
               else f" ±{res['observed_false_mattered_se']:.3f}")
    print(f"  FALSE `mattered`    : predicted "
          f"{res['predicted_false_mattered_rate']:.3f}   observed "
          f"{res['observed_false_mattered_rate']:.3f}{_se_txt}   "
          f"(formula error {res['formula_error']:+.3f})")
    print(f"     the ± is over {res.get('n_independent_units')} EPISODES, "
          f"not over splits: an episode's splits share legs, so the "
          f"episode is the unit of independence (measured, the two come "
          f"out close — the splits within an episode are heterogeneous)")
    fpd = res.get("false_mattered_given_decided")
    print(f"  …of the DECIDED ones: "
          f"{'—' if fpd is None else f'{fpd:.3f}'}  ← the number a "
          f"consumer of `mattered_*` is actually exposed to")
    print("  Every arm here is the SAME condition, so every `mattered_*` "
          "is a false positive by construction.")
    if res["verdict"] == "MEASURED":
        print(f"  VERDICT: MEASURED over {res['n_episodes']} episodes.")
        print("  What it licenses: `noise_floor` is already on every credit "
              "row. Where the formula tracks observation, that field is a "
              "validated PER-ROW trust score — so `dream_credit` can open "
              "conditionally on it rather than on a blanket pass.")
    elif res["verdict"] == "INCOMPLETE":
        print("  VERDICT: INCOMPLETE — the run did not finish.")
    else:
        print(f"  VERDICT: UNDERPOWERED — {MIN_EPISODES} episodes are "
              f"needed before this is a corpus statistic.")


def _install_stop_handlers() -> list:
    """Make SIGINT and SIGTERM raise, so a stop reaches the `finally`.

    ⚠ MEASURED. This run is launched in the background, and a shell
    starting a background job sets SIGINT to **SIG_IGN** in the child —
    which CPython PRESERVES rather than replacing with its default
    handler. `kill -INT` was therefore a no-op: the census kept walking
    the corpus and wrote no completion footer, which is exactly the case
    the footer exists for. SIGTERM has no default handler at all, so it
    kills the process outright and loses the same footer.

    Installing both explicitly overrides whatever was inherited, so an
    operator stopping a ten-hour run keeps every row it earned.
    """
    import signal as _signal

    state = {"stopping": False}

    def _stop(signum, _frame):
        # ⚠ ONE-SHOT. The `finally` that writes the footer runs with this
        # handler still armed, and the container teardown ahead of it
        # routinely takes seconds — so an operator who signals, sees the
        # process still alive, and signals again would raise INSIDE the
        # footer write and destroy the very footer this exists for.
        # Reproduced: second SIGTERM 0.6 s later → no footer at all.
        if state["stopping"]:
            return
        state["stopping"] = True
        raise KeyboardInterrupt(f"signal {signum}")

    installed = []
    for _sig in (_signal.SIGINT, _signal.SIGTERM):
        try:
            _signal.signal(_sig, _stop)
            installed.append(_sig)
        except Exception:               # noqa: BLE001
            pass
    return installed


def main() -> int:
    _install_stop_handlers()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--legs", type=int, default=DEFAULT_LEGS)
    ap.add_argument("--leg-timeout", type=float, default=240.0)
    ap.add_argument("--max-hours", type=float, default=8.0)
    ap.add_argument("--upstream", default="http://127.0.0.1:8088")
    ap.add_argument("--fork", choices=("live", "empty"), default="live",
                    help="the regime legs run in. `live` copies the live "
                         "sandbox into each fork, which is what the "
                         "engine's own run_spec does; `empty` starts "
                         "from nothing. Recorded on every run.")
    ap.add_argument("--only", default="",
                    help="comma-separated trajectory ids (prefixes ok) to "
                         "measure INSTEAD of walking the corpus. The run "
                         "records the subset in its header: a targeted "
                         "re-measure is not a corpus figure and must not "
                         "be rescored as one.")
    ap.add_argument("--home", default="")
    ap.add_argument("--no-vector", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rescore", default="",
                    help="score an EARLIER run's aa_rows.jsonl and exit")
    args = ap.parse_args()

    if args.rescore:
        if not Path(args.rescore).is_file():
            print(f"no such rows file: {args.rescore}", file=sys.stderr)
            return EXIT_NO_DATA
        _out = rescore(args.rescore)
        _res = score(_out)
        if args.json:
            print(json.dumps({"result": _res, "detail": _out}, indent=1))
        else:
            render(_out, _res)
        return EXIT_OK if _res["verdict"] == "MEASURED" else EXIT_INCOMPLETE

    if args.legs % 2 or args.legs < 4:
        print("--legs must be even and at least 4 (two arms of >=2)",
              file=sys.stderr)
        return EXIT_NO_DATA
    from ghost_agent.core.replay_engine import DEFAULT_N_PAIRS
    if args.legs // 2 != DEFAULT_N_PAIRS:
        # ⚠ The floor depends on n: at p=0.5 it is 0.333 with arms of 2
        # and 0.100 with arms of 3. Measuring a configuration the engine
        # does not run is measuring a system nobody uses — D4's own
        # review found exactly this and it cost a five-hour run.
        print(f"REFUSING: --legs {args.legs} gives arms of "
              f"{args.legs // 2}, but the engine ships DEFAULT_N_PAIRS="
              f"{DEFAULT_N_PAIRS}. The floor is a function of n.",
              file=sys.stderr)
        return EXIT_NO_DATA

    live = os.getenv("GHOST_HOME", "").strip()
    if not live or not Path(live).is_dir():
        print("GHOST_HOME is not set to a real directory", file=sys.stderr)
        return EXIT_NO_DATA

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from dream_replay_validate import claim_scratch, stamp_scratch
    from dream_replay_smoke import _build_context, _stage_home
    scratch = (Path(args.home) if args.home
               else Path(os.getenv("TMPDIR", "/tmp")) / "ghost-replay-aa")
    why = claim_scratch(scratch, Path(live))
    if why:
        print(f"REFUSING: {why}", file=sys.stderr)
        return EXIT_NO_DATA
    if not stamp_scratch(scratch):
        print(f"REFUSING: {scratch} could not be claimed", file=sys.stderr)
        return EXIT_NO_DATA
    if not args.home or not (scratch / "system" / "memory").is_dir():
        print(f"staging a COPY of {live} → {scratch}", flush=True)
        _stage_home(Path(live), scratch)
    for tree in ("system/trajectories", "system/counterfactual"):
        s, d = Path(live) / tree, scratch / tree
        if s.exists():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(s, d, dirs_exist_ok=True)
    os.environ["GHOST_HOME"] = str(scratch)

    from ghost_agent.core.replay_engine import preflight
    ok, why = preflight()
    print(f"preflight: {'CLEAR' if ok else 'BLOCKED'} — {why}", flush=True)
    if not ok:
        return EXIT_NO_DATA

    rows_path = scratch / "aa_rows.jsonl"
    print(f"per-episode rows → {rows_path}", flush=True)
    ctx = _build_context(scratch, args.upstream,
                         with_vector=not args.no_vector)
    out = None
    try:
        out = asyncio.run(measure(
            ctx, episodes=args.episodes, legs=args.legs,
            leg_timeout_s=args.leg_timeout,
            seeded_forks=(args.fork == "live"),
            only=({x.strip() for x in args.only.split(",") if x.strip()}
                  or None),
            run_deadline=time.monotonic() + args.max_hours * 3600.0,
            rows_path=rows_path))
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    except Exception as exc:                # noqa: BLE001
        print(f"\nrun FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        try:
            (scratch / ".d4-owner.json").unlink()
        except Exception:
            pass
    if out is None:
        print(f"no result — partial rows at {rows_path}", file=sys.stderr)
        return EXIT_INCOMPLETE
    res = score(out)
    if args.json:
        print(json.dumps({"result": res, "detail": out}, indent=1))
    else:
        render(out, res)
    return EXIT_OK if res["verdict"] == "MEASURED" else EXIT_INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main())
