#!/usr/bin/env python3
"""§4CM D4 — does the replay engine measure what it claims?

THE GATE BEFORE ANY CONSUMER. The engine's output will be treated as
LABELS, and this project's worst documented label failures — §4AO (52% of
skill-prune victims noise-decided), §4BE (59 false positives out of 59) —
were both plausible mechanisms with no sensitivity/specificity number
attached. This script is that number.

HOW THE SEEDS ARE CONSTRUCTED, and why it is not the obvious way.

`$GHOST_HOME/system/counterfactual/challenges.jsonl` holds self-play
challenges that each carry a `setup_script` building their fixture and a
`validation_script` that is REAL executable ground truth — no synthesis,
no model opinion. 173 of them additionally have a recorded `stable-pass`
verdict, i.e. a later replay reproduced the original success, so they are
known solvable, and every one asks for a `solution.py` that its validator
then runs with `subprocess` (checked per row, not assumed).

  * **POSITIVE** — ablate `file_system` AND `execute` for the whole
    perturbed leg. With both gone there is no path to create that file,
    so the validator's `subprocess.run(["python3","solution.py"])` MUST
    fail, whatever the model tries. Mechanically guaranteed, and getting
    there exercises the fork, the setup, the tool restriction, the agent
    run, the validator and the paired verdict rule.
  * **NULL** — ablate `recall` and `list_lessons`, tools the task never
    touches, through the IDENTICAL code path. The old seeded pair could
    not claim that: its null was a lesson withhold and its positive was a
    validator wrapper, so the two shared almost no machinery.
  * **STABILITY** — the same null spec, twice.

⚠ WHAT THIS REPLACED, and why. The first version wrapped the validator so
its exit code flipped. That measured the validator-execution path and
nothing else: the inversion is applied DOWNSTREAM of the agent run, so
the arms differ no matter what the perturbation did — the instrument
could not distinguish a working engine from one whose perturbations were
all inoperative. Three review rounds then found exactly that class of
defect (a withhold that did not withhold, a step-deny that never fired, a
control arm that was not the recorded condition), and the gate written to
catch it was structurally blind to every one. It was also confounded with
the wrong sign — an effective perturbation made the perturbed leg fail,
the inversion flipped it to pass, and a working engine scored a MISS.

⚠ WHAT THIS GATE DOES NOT CERTIFY. `tool_ablate` is a SEEDED
perturbation: `build_specs` never plans one, so this is a spike-in
control, not a sample of the engine's live workload. It certifies the
paired-verdict machinery — fork, restriction, run, validator, decision —
against a known effect of size 1.0. It says nothing about how often the
three MINED kinds (`lesson_withhold`, `step_deny`, `verify_toggle`) find
an effect, or about their applicability; that is D3's `unapplied` rate,
and it is a different number.

PRE-REGISTERED BARS (IDE.md §6 P4/P5): sensitivity >= 0.80,
specificity >= 0.90, flip rate <= 0.10. Below any of them, Dream stays a
producer of REPORTS and `dream_credit` stays closed.

Three honesty rules the script holds itself to:

  * An ABSTAIN is neither a hit nor a miss, and WHY it abstained is
    recorded per case. A control leg that ERRORED (docker, timeout, fork,
    setup) and a control leg that HONESTLY FAILED look identical in a
    sensitivity number and mean opposite things — one is an
    infrastructure fault, the other is "the task was not solved, so there
    was nothing to break". A perturbation that did not APPLY is a third
    thing again, and it is the failure this gate exists to catch.
  * A bar cannot be cleared without POWER **or** without a
    REPRESENTATIVE sample. Decided cases are the subset of tasks the
    agent could solve, so a sensitivity computed over 10% of the seeds is
    conditioned on the easy tail — hence `MIN_DECIDED_FRACTION` alongside
    `MIN_CASES_PER_ARM`. §4CE applied to the instrument that exists to
    prevent §4AO.
  * A run that stopped early, crashed, or ran out of its deadline cannot
    return PASS, however good the partial numbers look.

Runs against a COPY of `$GHOST_HOME`; the live agent holds the Chroma DB.
The copy is refused if it would resolve inside the live home, and it is
stamped with this process's PID so a second run cannot rsync over one in
flight.

    GHOST_HOME=... PYTHONPATH=src python scripts/dream_replay_validate.py \
        --cases 28 --stability-cases 18

Exit: 0 = every bar cleared; 1 = a bar missed; 2 = no data / not runnable;
3 = underpowered or unrepresentative; 4 = incomplete (stopped early).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BAR_SENSITIVITY = 0.80
BAR_SPECIFICITY = 0.90
BAR_FLIP_RATE = 0.10
#: Smallest seeded-case count per arm that can distinguish the bar from
#: chance. Exact binomial, two-sided alpha 0.05: at n=10 the power to
#: reject "sensitivity = 0.5" when the truth is 0.8 is 0.53; at n=20 it
#: is 0.87. Below 20 the script refuses a PASS.
MIN_CASES_PER_ARM = 20
#: AMENDMENT to §6's pre-registration, added after review. `powered` on
#: its own is an ABSOLUTE count, so an engine that abstains on 90% of
#: seeded positives and is right about the other 10% clears it with
#: sensitivity 1.00 — demonstrated, not argued. Two things are wrong with
#: that: the yield itself is a defect the bars cannot see, and the
#: decided cases are exactly the tasks the agent could solve, so the
#: estimate is conditioned on the easy tail. A gate whose sample is that
#: unrepresentative reports UNREPRESENTATIVE, never PASS.
MIN_DECIDED_FRACTION = 0.50
#: The flip-rate bar had no floor at all: `stability n=1, flips=0` cleared
#: it. Ten decided repeats can only exclude a flip rate above 0.259 (exact
#: upper bound at 0/10), which the render prints so nobody mistakes a
#: cleared bar for a tight estimate.
MIN_STABILITY_N = 10
#: Observed probability that a stability REPEAT decides — it counts only
#: when BOTH the null run and its repeat decide. Measured on a live run:
#: 9 attempted, 6 decided.
STABILITY_DECIDE_RATE = 0.67
#: Attempts needed for `MIN_STABILITY_N` decided repeats with ~85% power,
#: not merely in expectation. ⚠ The first version had TWO formulas — a
#: warning threshold of `MIN_STABILITY_N * 1.5` and a recommendation of
#: `MIN_STABILITY_N / 0.67 + 3` — which coincide only at
#: `MIN_STABILITY_N == 10`, so the warning was silent at exactly the
#: default it shipped with, and the default itself (15) had a 0.63
#: chance of reaching the floor: a one-in-three chance of returning
#: UNDERPOWERED after eight hours. One formula, used by the default, the
#: warning and the test alike.
STABILITY_ATTEMPTS = 18
#: Wall-clock a spec needs beyond its legs: fork + provision + setup exec
#: + validator exec + teardown, per leg. `run_spec`'s default deadline is
#: a flat 1500 s, which is SHORTER than 6 legs at the default leg timeout
#: — and control legs run first, so the shortfall lands entirely on the
#: perturbed arm and starves one arm of the comparison.
SPEC_OVERHEAD_PER_LEG_S = 90.0

EXIT_PASS = 0
EXIT_MISS = 1
EXIT_NO_DATA = 2
EXIT_UNDERPOWERED = 3
EXIT_INCOMPLETE = 4

#: Why a run stopped short. Both are FATAL to a verdict, and the second
#: one is a correction.
#:
#: ⚠ AN EARLIER VERSION ARGUED THAT A WALL-CLOCK TRUNCATION WAS STILL
#: EVALUABLE, on the grounds that the corpus is shuffled and whole
#: UNSTARTED cases are what a deadline drops, so what ran is a uniform
#: random subsample. **That is wrong**, and the error is worth keeping
#: written down: a uniform random ORDER does not give a uniform random
#: RETAINED SET when the stopping rule is duration-dependent. The
#: retained set is the maximal prefix that fits in T, so a case's
#: probability of completing FALLS with its duration — the classic
#: inspection/knapsack bias. And duration here is strongly
#: outcome-correlated: a case the agent flounders on burns
#: `n_pairs × leg_timeout` per arm while one it solves in 30 s costs a
#: fraction of that. Hard cases are systematically under-represented,
#: and "hard" is exactly "control arm fails or abstains" — the
#: conditioning `MIN_DECIDED_FRACTION` exists to police. Worse, the
#: shuffle seed is fixed, so the same biased prefix is drawn every run
#: and the bias never averages out.
STOP_PREFLIGHT = "preflight"
STOP_DEADLINE = "deadline"

VERDICT_EXIT = {
    "PASS": EXIT_PASS,
    "BELOW BAR": EXIT_MISS,
    "NO DATA": EXIT_NO_DATA,
    "UNDERPOWERED": EXIT_UNDERPOWERED,
    "UNREPRESENTATIVE": EXIT_UNDERPOWERED,
    "INCOMPLETE": EXIT_INCOMPLETE,
}


# ── Seeded truth, from challenges that carry their own ground truth ──
#
# ⚠ WHY NOT THE INVERTED VALIDATOR. The first version paired a normal
# control leg with a leg whose validator had been WRAPPED to flip its
# exit code. That measured the validator-execution path and nothing else:
# the inversion is applied DOWNSTREAM of the agent run, so the arms differ
# no matter what the perturbation did — which means the instrument could
# not distinguish a working engine from one whose perturbations are all
# inoperative. Three review rounds found exactly that class of defect
# (a withhold that did not withhold, a step-deny that never fired, a
# control arm that was not the recorded condition), and the gate written
# to catch it was structurally blind to every one.
#
# It was also confounded with the wrong sign: the perturbed arm applied
# the REAL perturbation too, so an effective one made that leg fail, the
# inversion flipped it to pass, both arms read "passed", and a working
# engine scored a MISS. And a validator that merely crashed scored a HIT.
#
# WHAT REPLACES IT. `$GHOST_HOME/system/counterfactual/challenges.jsonl`
# holds 319 self-play challenges, each with a `setup_script` that builds
# its fixture and a `validation_script` that is REAL executable ground
# truth — no synthesis, no model opinion. 173 of them have a recorded
# `stable-pass` verdict, i.e. they are known solvable.
#
# So: ablate `file_system` AND `execute` for the whole perturbed leg.
# With both gone there is no path to create `solution.py`, so the
# validator's `subprocess.run(["python3", "solution.py"])` MUST fail —
# whatever the model does, whatever it decides to try. That is a
# mechanically guaranteed positive, and reaching it exercises the fork,
# the setup, the tool restriction, the agent run, the validator and the
# paired verdict rule: everything the inversion skipped.
#
# The matched NULL travels the identical code path — ablate tools the
# task never needed — which the old pair could not claim, because its
# null was a lesson withhold and its positive was a validator wrapper.

#: Tools a `solution.py` task cannot be completed without.
ESSENTIAL_TOOLS = "file_system,execute"
#: Tools such a task never touches. Ablating them must be inert.
IRRELEVANT_TOOLS = "recall,list_lessons"
#: The artefact the positive's guarantee rests on. ENFORCED per row
#: rather than assumed: 173/173 qualifying rows satisfy it today, but a
#: future challenge that validates stdout, or a pre-existing file, would
#: enter the seed set silently and break the guarantee — and the symptom
#: would be "the engine cannot detect perturbations".
_REQUIRED_ARTEFACT = "solution.py"

_CHALLENGE_FILE = "challenges.jsonl"
_RESULTS_FILE = "results.jsonl"

#: Why a case produced no usable comparison. These are NOT
#: interchangeable: the first is an infrastructure fault, the second is
#: the agent failing a solvable task, the third is the failure this gate
#: exists to catch, and folding them together is how "the engine cannot
#: detect" becomes indistinguishable from "docker was down".
ABSTAIN_CONTROL_ERROR = "control leg ERRORED (fork/setup/timeout/sandbox)"
ABSTAIN_CONTROL_FAILED = "control leg FAILED the task (nothing to break)"
ABSTAIN_CONTROL_SPLIT = "control legs DISAGREED (task is stochastic)"
ABSTAIN_UNAPPLIED = "perturbation did NOT APPLY"
ABSTAIN_PERT_ERROR = "perturbed leg ERRORED"
ABSTAIN_PERT_SPLIT = "perturbed legs DISAGREED (task is stochastic)"
ABSTAIN_RUN_ERROR = "the case raised"


def load_seed_challenges(home: str = None, limit: int = 40,
                         shuffle_seed: int = 0) -> list:
    """Challenges with a real setup, a real validator that runs
    `solution.py`, and a recorded `stable-pass` verdict — i.e. known
    solvable, so a control leg that fails is the engine's problem and not
    the task's.

    Sampled with a SEEDED shuffle rather than file order: the corpus is
    chronological, so the first N are the oldest N, and every re-run
    would measure the same head of the distribution with no held-out
    remainder to check a fix against.
    """
    base = (home or os.getenv("GHOST_HOME", "")).strip()
    if not base:
        return []
    d = Path(base) / "system" / "counterfactual"
    try:
        rows = [json.loads(l) for l in
                (d / _CHALLENGE_FILE).read_text().splitlines() if l.strip()]
    except Exception:
        return []
    verdicts = {}
    try:
        for line in (d / _RESULTS_FILE).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                verdicts[r.get("challenge_id")] = r.get("verdict")
    except Exception:
        pass
    out = []
    for r in rows:
        if not (r.get("setup_script") and r.get("validation_script")):
            continue
        if r.get("status") != "SUCCESS":
            continue
        # `stable-pass` means a later replay reproduced the original
        # SUCCESS — the strongest "this is solvable" evidence on disk.
        if verdicts.get(r.get("id")) != "stable-pass":
            continue
        # The positive's guarantee is "there is no path to the artefact
        # the validator runs". A validator that never mentions it is not
        # a seed this script can make that claim about.
        if _REQUIRED_ARTEFACT not in str(r.get("validation_script") or ""):
            continue
        if _REQUIRED_ARTEFACT in str(r.get("setup_script") or ""):
            continue
        out.append(r)
    random.Random(shuffle_seed).shuffle(out)
    return out[:max(0, int(limit))]


def seed_specs(challenge: dict, RE) -> tuple:
    """(positive, null) specs for one challenge, through one code path."""
    base = {
        "trajectory_id": str(challenge.get("id") or "")[:16],
        "user_request": str(challenge.get("challenge") or ""),
        "setup_script": str(challenge.get("setup_script") or ""),
        "recorded_outcome": "passed",
        "fork_step": 0,
        "perturbation": RE.PERTURB_TOOL_ABLATE,
        "n_steps": 0,
    }
    pos = dict(base, spec_id=f"seedpos-{base['trajectory_id']}",
               target=ESSENTIAL_TOOLS)
    null = dict(base, spec_id=f"seednull-{base['trajectory_id']}",
                target=IRRELEVANT_TOOLS)
    return pos, null


def classify(rec: dict, *, want: str, RE) -> tuple:
    """(outcome, reason) for one spec record, where outcome is
    ``hit`` | ``miss`` | ``abstain``.

    ⚠ THE ORDER OF THESE BRANCHES IS THE FIX. Reading `rec["verdict"]`
    first scores a case whose control arm never solved the task as a
    sensitivity MISS: both arms fail, `decide_verdict` correctly returns
    NO_EFFECT, and the gate charges an engine that did nothing wrong. The
    seed's ground truth is *conditional* — "if the task was solved, the
    ablation must break it" — so the condition is checked BEFORE the
    verdict is read, not after.
    """
    ctrl = list(rec.get("control_pass") or [])
    pert = list(rec.get("pert_pass") or [])
    if not ctrl or any(x is None for x in ctrl):
        return "abstain", ABSTAIN_CONTROL_ERROR
    if not any(ctrl):
        return "abstain", ABSTAIN_CONTROL_FAILED
    if not all(ctrl):
        return "abstain", ABSTAIN_CONTROL_SPLIT
    # Reached only with a clean, unanimous, PASSING control arm, so
    # `run_spec` ran the perturbed arm and an unapplied perturbation is a
    # genuine non-application rather than a control-side casualty.
    if rec.get("applied") is False:
        return "abstain", ABSTAIN_UNAPPLIED
    if not pert or any(x is None for x in pert):
        return "abstain", ABSTAIN_PERT_ERROR
    if len(set(pert)) > 1:
        return "abstain", ABSTAIN_PERT_SPLIT
    v = str(rec.get("verdict") or "")
    if v == want:
        return "hit", ""
    if v == RE.VERDICT_ABSTAIN:
        return "abstain", str(rec.get("why") or "")[:80]
    # A `mattered_neg` on the POSITIVE arm means control failed and the
    # ablated leg passed, which the branches above have excluded — so it
    # can only be a sign inversion in `decide_verdict`. Counting it as a
    # hit is how an instrument goes blind to the exact confound that
    # killed its predecessor.
    return "miss", ("SIGN INVERTED" if v == RE.VERDICT_MATTERED_NEG else v)


def _blank_arm() -> dict:
    return {"n": 0, "hit": 0, "miss": 0, "abstain": 0, "why": {}}


def _tally(arm: dict, outcome: str, reason: str) -> None:
    arm["n"] += 1
    arm[outcome] += 1
    if outcome != "hit" and reason:
        arm["why"][reason] = arm["why"].get(reason, 0) + 1


async def validate(context, *, cases: int = 20, n_pairs: int = None,
                   stability_cases: int = STABILITY_ATTEMPTS,
                   leg_timeout_s: float = 300.0,
                   run_deadline: float = None,
                   shuffle_seed: int = 0,
                   rows_path=None) -> dict:
    """Run the seeded pairs and report what the engine did with them.

    ``n_pairs`` defaults to the ENGINE's own default. A gate that
    certifies a cheaper configuration than the one that ships is
    certifying a system nobody runs — the same shape as measuring
    precision over a population the steer never touches (§4CL R2 M3).
    """
    from ghost_agent.core import replay_engine as RE

    if n_pairs is None:
        n_pairs = RE.DEFAULT_N_PAIRS

    out = {
        "positives": _blank_arm(), "nulls": _blank_arm(),
        "stability": {"n": 0, "decided": 0, "flips": 0},
        "challenges_loaded": 0, "challenges_run": 0,
        "rows": [], "stopped_early": "", "stop_cause": "", "reason": "",
        "n_pairs": n_pairs, "errors": 0,
    }

    challenges = load_seed_challenges(limit=cases, shuffle_seed=shuffle_seed)
    out["challenges_loaded"] = len(challenges)
    if not challenges:
        out["reason"] = ("no seed challenges — "
                         "system/counterfactual/challenges.jsonl is empty or "
                         "has no stable-pass rows whose validator runs "
                         f"{_REQUIRED_ARTEFACT}")
        return out

    # A spec's deadline has to hold the legs it is about to run. The
    # engine's flat 1500 s default is shorter than 2*n_pairs legs at the
    # default leg timeout, and control legs run FIRST, so the whole
    # shortfall lands on the perturbed arm: the positive (3 slow control
    # legs + 3 fast crippled ones) fits, the null (6 full-capability
    # legs) does not, and specificity ends up estimated on the fastest
    # subset while sensitivity is estimated on all of them. That is a
    # difference between the arms that is not the tool list.
    spec_timeout_s = (2 * max(1, int(n_pairs))
                      * (float(leg_timeout_s) + SPEC_OVERHEAD_PER_LEG_S))

    from ghost_agent.core.replay_engine import preflight as _pf

    # ⚠ EVERY ROWS FILE CARRIES A RUN ID. `_emit` appends, the default
    # scratch path is fixed, and "re-run after a fix" is the documented
    # workflow — so a second run's rows landed in the first run's file
    # and `--rescore` counted A+B: the same challenge twice, across two
    # different configurations, with nothing on the rows to say so.
    _run_id = f"{os.getpid()}-{int(time.time())}"
    if rows_path:
        try:
            with open(rows_path, "a") as fh:
                fh.write(json.dumps({
                    "__run__": "start", "run_id": _run_id,
                    "n_pairs": n_pairs, "cases": cases,
                    "stability_cases": stability_cases,
                    "leg_timeout_s": leg_timeout_s,
                    "shuffle_seed": shuffle_seed}) + "\n")
        except Exception:
            pass

    def _emit(row):
        row["run_id"] = _run_id
        out["rows"].append(row)
        if rows_path:
            # A five-hour measurement that lives only in RAM is one
            # unhandled exception away from never having happened.
            try:
                with open(rows_path, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
            except Exception:
                pass

    async def _run(spec, validator):
        # ⚠ NO `batch_deadline`. Passing the run deadline through clamps
        # the SPEC deadline, so the case in flight when the clock runs
        # out is not dropped — it is CORRUPTED: `run_leg` squeezes or
        # skips its remaining legs, and the resulting abstain is filed as
        # "fork/setup/timeout/sandbox", or "the task was not solved", or
        # "the task is stochastic". Three different lies at exactly the
        # boundary. It is also arm-asymmetric, because the positive spec
        # runs first. The deadline is checked BETWEEN cases instead, so
        # a run overshoots by at most one case and every abstain means
        # what it says.
        return await RE.run_spec(
            context, spec, validator=validator, source_workspace=None,
            n_pairs=n_pairs, leg_timeout_s=leg_timeout_s,
            spec_timeout_s=spec_timeout_s, write=False)

    for ch in challenges:
        if run_deadline is not None and time.monotonic() >= run_deadline:
            out["stopped_early"] = "run deadline reached"
            out["stop_cause"] = STOP_DEADLINE
            print(f"  STOPPING: {out['stopped_early']}", flush=True)
            break
        # Resources change under a run that spawns a container per leg,
        # and this one is hours long. A single reading at entry is not a
        # guarantee — the same lesson `run_batch` carries.
        _ok, _why = _pf()
        if not _ok:
            out["stopped_early"] = f"preflight stood down mid-run: {_why}"
            out["stop_cause"] = STOP_PREFLIGHT
            print(f"  STOPPING: {_why}", flush=True)
            break
        pos, null = seed_specs(ch, RE)
        validator = str(ch.get("validation_script") or "")
        row = {"id": pos["trajectory_id"], "cluster": ch.get("cluster")}
        out["challenges_run"] += 1

        try:
            # ── The POSITIVE. Both arms run; the perturbed one has no way
            # to produce `solution.py`, so it must fail.
            rec = await _run(pos, validator)
            p_out, p_why = classify(rec, want=RE.VERDICT_MATTERED_POS, RE=RE)
            row.update(positive=rec["verdict"], positive_outcome=p_out,
                       positive_why=p_why, pos_control=rec["control_pass"],
                       pos_pert=rec["pert_pass"], pos_applied=rec["applied"])
            _tally(out["positives"], p_out, p_why)

            # ── The matched NULL, through the identical code path.
            nrec = await _run(null, validator)
            n_out, n_why = classify(nrec, want=RE.VERDICT_NO_EFFECT, RE=RE)
            row.update(null=nrec["verdict"], null_outcome=n_out,
                       null_why=n_why, null_pass_rate=nrec.get("pass_rate"),
                       null_noise_floor=nrec.get("noise_floor"))
            _tally(out["nulls"], n_out, n_why)

            # ── STABILITY: the same NULL spec again. Only a repeat where
            # BOTH runs DECIDED can flip — counting abstain==abstain as
            # agreement gives an engine that abstains on everything a
            # perfect stability score.
            if out["stability"]["n"] < stability_cases:
                again = await _run(null, validator)
                a_out, _ = classify(again, want=RE.VERDICT_NO_EFFECT, RE=RE)
                out["stability"]["n"] += 1
                if n_out != "abstain" and a_out != "abstain":
                    out["stability"]["decided"] += 1
                    if again["verdict"] != nrec["verdict"]:
                        out["stability"]["flips"] += 1
                row["stability"] = [nrec["verdict"], again["verdict"]]
                row["stability_outcomes"] = [n_out, a_out]
        except Exception as exc:               # noqa: BLE001
            # One bad case must not discard the hours already spent.
            out["errors"] += 1
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
            # ⚠ ONTO THE ROW, not only into the tally. `rescore` reads
            # rows, so a case whose outcome existed only in memory
            # vanished from `n`, from `abstain` and from the `why`
            # histogram on recovery — losing the infra-fault taxonomy AND
            # inflating the decided fraction in the PASS direction.
            for arm, key in (("positives", "positive"), ("nulls", "null")):
                if not row.get(f"{key}_outcome"):
                    row[f"{key}_outcome"] = "abstain"
                    row[f"{key}_why"] = ABSTAIN_RUN_ERROR
                    _tally(out[arm], "abstain", ABSTAIN_RUN_ERROR)

        _emit(row)
        print(f"  {row['id']:16} positive={row.get('positive','—'):13}"
              f"[{row.get('positive_outcome','—'):7}] "
              f"null={row.get('null','—'):13}"
              f"[{row.get('null_outcome','—'):7}] "
              f"pass_rate={row.get('null_pass_rate')}", flush=True)
    if rows_path:
        # The footer is what lets `--rescore` tell a COMPLETE run from a
        # killed one. Without it a rescore could only guess, and guessing
        # in the optimistic direction is how a partial run prints PASS.
        try:
            with open(rows_path, "a") as fh:
                fh.write(json.dumps({
                    "__run__": "end", "run_id": _run_id,
                    "stopped_early": out["stopped_early"],
                    "stop_cause": out["stop_cause"],
                    "challenges_loaded": out["challenges_loaded"],
                    "challenges_run": out["challenges_run"]}) + "\n")
        except Exception:
            pass
    return out


# ── Exact (Clopper-Pearson) interval. §4CE: a point estimate that clears
# a bar with an interval straddling it is not evidence the bar is
# cleared, and this project has already shipped ten arms whose "no
# difference" verdict described an undetectable difference. The verdict
# rule stays as PRE-REGISTERED (point estimate + power); the interval is
# printed so the reader can see how much of it is luck.
def _binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
               for i in range(k + 1))


def _solve(fn, target: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Bisect a monotone fn for fn(p) == target (40 halvings ⇒ ~1e-12)."""
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if (fn(mid) > target) == (fn(lo) > target):
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2.0, 4)


def ci95(k: int, n: int) -> tuple:
    """Exact two-sided 95% interval for k successes in n trials."""
    if n <= 0:
        return (None, None)
    lo = 0.0 if k <= 0 else _solve(
        lambda p: 1.0 - _binom_cdf(k - 1, n, p), 0.025)
    hi = 1.0 if k >= n else _solve(lambda p: _binom_cdf(k, n, p), 0.025)
    return (lo, hi)


def score(out: dict) -> dict:
    p, n, st = out["positives"], out["nulls"], out["stability"]
    # Abstains are their OWN column. Folding them into either bucket is
    # how an engine that abstains on nine of ten seeded positives reports
    # a sensitivity of 1.00.
    p_dec, n_dec = p["hit"] + p["miss"], n["hit"] + n["miss"]
    st_dec = st["decided"]
    # ⚠ DENOMINATOR = THE SEEDS, NOT THE PREFIX. `p["n"]` counts cases
    # ATTEMPTED, so a run that stopped after 21 of 173 with all 21
    # decided reported a yield of 1.00 and `representative=True` while
    # 88% of the corpus was never measured. The seeds are the population
    # the render's own sentence claims to describe.
    _loaded = max(int(out.get("challenges_loaded") or 0), p["n"], n["n"])
    p_yield = (p_dec / _loaded) if _loaded else None
    n_yield = (n_dec / _loaded) if _loaded else None
    res = {
        "sensitivity": (p["hit"] / p_dec) if p_dec else None,
        "specificity": (n["hit"] / n_dec) if n_dec else None,
        "sensitivity_ci95": ci95(p["hit"], p_dec),
        "specificity_ci95": ci95(n["hit"], n_dec),
        "flip_rate": (st["flips"] / st_dec) if st_dec else None,
        "flip_rate_ci95": ci95(st["flips"], st_dec),
        # `n_*` used to be the ATTEMPTED count sitting in the same dict as
        # a ratio computed over the DECIDED count — a --json consumer read
        # "sensitivity 1.0, n 200" off 20 supporting cases.
        "n_positives_decided": p_dec, "n_nulls_decided": n_dec,
        "n_positives_attempted": p["n"], "n_nulls_attempted": n["n"],
        "n_stability_decided": st_dec, "n_stability_attempted": st["n"],
        "positive_decided_fraction": p_yield,
        "null_decided_fraction": n_yield,
        "decided_fraction_denominator": _loaded,
        "positive_abstain_rate": (p["abstain"] / p["n"]) if p["n"] else None,
        "null_abstain_rate": (n["abstain"] / n["n"]) if n["n"] else None,
        # ⚠ THE REGIME THE NUMBERS WERE MEASURED IN. The paired rule's
        # false-`mattered` rate is 2·pⁿ·qⁿ/(pⁿ+qⁿ)² — ZERO when the task
        # is deterministic. The seed corpus is `stable-pass` self-play
        # challenges, so p=1.0 on nearly every case, which means the
        # specificity and flip-rate arms are measured where they CANNOT
        # fail. A reader comparing them to the live corpus (whose control
        # legs agreed with the recording 72.7% of the time) is comparing
        # two different experiments.
        "measured_at_pass_rate": _regime(out),
        "bars": {"sensitivity": BAR_SENSITIVITY,
                 "specificity": BAR_SPECIFICITY,
                 "flip_rate": BAR_FLIP_RATE,
                 "min_cases_per_arm": MIN_CASES_PER_ARM,
                 "min_stability_n": MIN_STABILITY_N,
                 "min_decided_fraction": MIN_DECIDED_FRACTION},
    }
    powered = (p_dec >= MIN_CASES_PER_ARM and n_dec >= MIN_CASES_PER_ARM
               and st_dec >= MIN_STABILITY_N)
    representative = (p_yield is not None and n_yield is not None
                      and p_yield >= MIN_DECIDED_FRACTION
                      and n_yield >= MIN_DECIDED_FRACTION)
    res["powered"] = powered
    res["representative"] = representative
    res["missed"] = []
    if out.get("reason") and not out["rows"]:
        res["verdict"] = "NO DATA"
        return res
    # A partial run cannot certify anything, however good its numbers
    # look — checked BEFORE the bars so no ordering accident can let an
    # interrupted run print PASS. EVERY stop cause is fatal, including
    # the wall-clock one: see STOP_DEADLINE's note, the "it is still a
    # uniform sample" argument does not survive the fact that slow cases
    # are the ones that get dropped and slow means hard.
    res["truncated"] = bool(out.get("stopped_early"))
    if out.get("stopped_early"):
        res["verdict"] = "INCOMPLETE"
        return res
    if not powered:
        res["verdict"] = "UNDERPOWERED"
        return res
    if not representative:
        res["verdict"] = "UNREPRESENTATIVE"
        return res
    misses = []
    if res["sensitivity"] is None or res["sensitivity"] < BAR_SENSITIVITY:
        misses.append("sensitivity")
    if res["specificity"] is None or res["specificity"] < BAR_SPECIFICITY:
        misses.append("specificity")
    if res["flip_rate"] is None or res["flip_rate"] > BAR_FLIP_RATE:
        misses.append("flip_rate")
    res["missed"] = misses
    res["verdict"] = "PASS" if not misses else "BELOW BAR"
    return res


def rescore(rows_path) -> dict:
    """Rebuild the whole scoreboard from `d4_rows.jsonl`.

    A run that spans hours WILL sometimes end in a way its own `score`
    call never sees — a kill, a crash, an operator's ^C, a machine
    reboot. Streaming the rows to disk only helps if something can read
    them back, so this is the other half of that fix.

    ⚠ TWO THINGS IT MUST NOT DO, both of which it did.
    (1) Return PASS for a killed run. It hardcoded `stopped_early: ""`,
    so `score`'s fatal branch was never even entered and the documented
    recovery command turned an EXIT_INCOMPLETE run into exit 0. A rows
    file is now only complete if it carries the writer's own `__run__:
    end` footer; without one, the run did not finish, full stop.
    (2) Merge two runs. `_emit` appends and the scratch path is fixed, so
    a second run's rows landed in the first run's file and this counted
    A+B — the same challenge twice, across two configurations. Rows now
    carry a `run_id` and only the LAST run's rows are scored.
    """
    raw = []
    for line in Path(rows_path).read_text().splitlines():
        if line.strip():
            try:
                raw.append(json.loads(line))
            except Exception:
                continue
    markers = [r for r in raw if r.get("__run__")]
    run_ids = [r.get("run_id") for r in markers if r.get("run_id")]
    last_run = run_ids[-1] if run_ids else None
    rows = [r for r in raw if not r.get("__run__")
            and (last_run is None or r.get("run_id") == last_run)]
    dropped = len([r for r in raw if not r.get("__run__")]) - len(rows)
    footer = header = None
    if last_run is not None:
        footer = next((r for r in reversed(markers)
                       if r.get("__run__") == "end"
                       and r.get("run_id") == last_run), None)
        header = next((r for r in reversed(markers)
                       if r.get("__run__") == "start"
                       and r.get("run_id") == last_run), None)
    out = {
        "positives": _blank_arm(), "nulls": _blank_arm(),
        "stability": {"n": 0, "decided": 0, "flips": 0},
        "challenges_loaded": len(rows), "challenges_run": len(rows),
        "rows": rows, "stopped_early": "", "stop_cause": "", "reason": "",
        "n_pairs": None, "errors": 0, "rescored_from": str(rows_path),
        "rescored_run_id": last_run, "rescored_dropped_rows": dropped,
    }
    if header:
        out["n_pairs"] = header.get("n_pairs")
        out["cases_requested"] = header.get("cases")
    if last_run is None:
        # ⚠ GATE ON `run_id`, NOT ON `markers`. Keying the legacy branch
        # on "are there markers" left a third class silent: markers that
        # carry no `run_id` (a build between the two fixes, or two files
        # concatenated). `last_run` was then None, the row filter
        # degraded to "every row", and the footer lookup matched via
        # `None == None` — so two runs merged into one scoreboard, with
        # `rescored_dropped_rows` reporting 0, `stopped_early` empty, and
        # the verdict PASS at exit 0. Both failures the docstring above
        # says this must not do, in one file.
        out["stopped_early"] = ("the rows file has no usable run id — it "
                                "pre-dates the markers, or holds rows from "
                                "runs that cannot be told apart")
    elif footer is None:
        out["stopped_early"] = ("no completion footer in the rows file — "
                                "the run did not finish (killed, crashed, "
                                "or interrupted)")
    else:
        out["stopped_early"] = str(footer.get("stopped_early") or "")
        out["stop_cause"] = str(footer.get("stop_cause") or "")
        out["challenges_loaded"] = int(footer.get("challenges_loaded")
                                       or out["challenges_loaded"])
        out["challenges_run"] = int(footer.get("challenges_run")
                                    or len(rows))
    for r in rows:
        if r.get("error"):
            out["errors"] += 1
        for arm, key in (("positives", "positive"), ("nulls", "null")):
            outcome = r.get(f"{key}_outcome")
            if outcome in ("hit", "miss", "abstain"):
                _tally(out[arm], outcome, r.get(f"{key}_why") or "")
            elif outcome:
                # A value nothing wrote is a corrupt row, not a category.
                # `_tally` would have raised KeyError on the one input
                # this function is guaranteed to be reading: a file
                # written by a process that died mid-line.
                _tally(out[arm], "abstain", f"unreadable outcome {outcome!r}")
        st = r.get("stability") or []
        so = r.get("stability_outcomes") or []
        if len(st) == 2:
            out["stability"]["n"] += 1
            if len(so) == 2 and "abstain" not in so:
                out["stability"]["decided"] += 1
                if st[0] != st[1]:
                    out["stability"]["flips"] += 1
    return out


def _regime(out: dict) -> dict:
    """Pass rates the null arm was actually measured at, and the noise
    floor they imply."""
    rates = [r.get("null_pass_rate") for r in out.get("rows") or []
             if r.get("null_pass_rate") is not None]
    floors = [r.get("null_noise_floor") for r in out.get("rows") or []
              if r.get("null_noise_floor") is not None]
    if not rates:
        return {"n": 0}
    det = sum(1 for r in rates if r >= 0.999)
    return {"n": len(rates), "deterministic": det,
            "deterministic_share": round(det / len(rates), 3),
            "median_pass_rate": sorted(rates)[len(rates) // 2],
            "max_noise_floor": max(floors) if floors else None}


def _pct(v) -> str:
    return "—" if v is None else f"{v:.3f}"


def _ci(pair) -> str:
    lo, hi = pair if pair else (None, None)
    return "" if lo is None else f" [{lo:.2f},{hi:.2f}]"


def _render(out: dict, res: dict) -> None:
    print("§4CM D4 — seeded-truth validation of the replay engine")
    print(f"  seed challenges: {out.get('challenges_run', 0)} run of "
          f"{out.get('challenges_loaded', 0)} loaded"
          f"   legs per arm: "
          f"{out.get('n_pairs') if out.get('n_pairs') else 'unknown'}")
    if out.get("reason"):
        print(f"  ⚠ {out['reason']}")
    if out.get("rescored_from"):
        print(f"  (rescored from {out['rescored_from']}, run "
              f"{out.get('rescored_run_id') or 'unknown'}"
              + (f"; {out['rescored_dropped_rows']} row(s) from EARLIER "
                 f"runs in the same file were ignored"
                 if out.get("rescored_dropped_rows") else "") + ")")
    if out.get("stopped_early"):
        # ⚠ This line used to append "a wall-clock truncation of a
        # SHUFFLED corpus, so what ran is a uniform random subsample" —
        # the exact claim `STOP_DEADLINE`'s note retracts. The retraction
        # reached `score`, the constant and the tests and stopped one
        # function short of the only surface an operator reads, leaving
        # two adjacent lines saying opposite things about one run.
        print(f"  ⚠ STOPPED EARLY: {out['stopped_early']}")
    if out.get("errors"):
        print(f"  ⚠ {out['errors']} case(s) raised and were counted as "
              f"abstains, not as successes")
    for arm, bar, key in (("positives", BAR_SENSITIVITY, "sensitivity"),
                          ("nulls", BAR_SPECIFICITY, "specificity")):
        a = out[arm]
        print(f"  {arm:10} n={a['n']:3}  hit={a['hit']:3} miss={a['miss']:3} "
              f"ABSTAIN={a['abstain']:3}   "
              f"{key} {_pct(res[key])}{_ci(res[key + '_ci95'])} "
              f"(bar {bar:.2f}, over {res['n_' + arm + '_decided']} decided)")
        # These are the difference between "the engine cannot detect" and
        # "the seed never reached the engine", and they look identical in
        # a sensitivity number.
        for why, cnt in sorted(a["why"].items(), key=lambda kv: -kv[1]):
            print(f"             {cnt:3}x {why}")
    st = out["stability"]
    print(f"  stability  n={st['n']:3}  decided={st['decided']:3} "
          f"flips={st['flips']:3}   flip rate {_pct(res['flip_rate'])}"
          f"{_ci(res['flip_rate_ci95'])} (bar {BAR_FLIP_RATE:.2f})")
    v = res["verdict"]
    if v == "NO DATA":
        print("  VERDICT: NO DATA — nothing ran.")
    elif v == "INCOMPLETE":
        print("  VERDICT: INCOMPLETE — the run did not finish, so its "
              "numbers describe whatever subset it got to. Not a result.")
    elif v == "UNDERPOWERED":
        print(f"  VERDICT: UNDERPOWERED — {MIN_CASES_PER_ARM} DECIDED cases "
              f"per arm and {MIN_STABILITY_N} decided repeats are needed "
              f"before a bar can be cleared. The estimates above are not a "
              f"result.")
    elif v == "UNREPRESENTATIVE":
        print(f"  VERDICT: UNREPRESENTATIVE — under "
              f"{MIN_DECIDED_FRACTION:.0%} of the "
              f"{res.get('decided_fraction_denominator')} seeded cases "
              f"produced a comparison (positives "
              f"{_pct(res['positive_decided_fraction'])}"
              f", nulls {_pct(res['null_decided_fraction'])}). The decided "
              f"cases are the ones the agent could solve, so these ratios "
              f"describe the easy tail, not the engine.")
    else:
        print(f"  VERDICT: {v}"
              + (f" — missed {', '.join(res['missed'])}"
                 if res.get("missed") else ""))
    reg = res.get("measured_at_pass_rate") or {}
    if reg.get("n"):
        print(f"  ⚠ REGIME: {reg['deterministic']}/{reg['n']} null cases ran "
              f"on DETERMINISTIC tasks (pass rate 1.0), where the paired "
              f"rule's false-`mattered` rate is 0.000 by construction. The "
              f"largest noise floor any case here saw was "
              f"{reg.get('max_noise_floor')}.")
        if reg.get("deterministic_share", 0) >= 0.75:
            print("     Specificity and flip rate above therefore describe a "
                  "regime the live corpus is NOT in — its control legs "
                  "agreed with the recording 72.7% of the time, and at "
                  "p=0.727 the floor is 0.096. Those two bars do not "
                  "transfer; the sensitivity bar does.")
    print("  Scope: this certifies that the paired-verdict MACHINERY is "
          "operative — fork, setup, tool restriction, agent run, "
          "validator, decision — against a SEEDED perturbation of effect "
          "size 1.0. Zero of the 309 specs the engine actually plans are "
          "this kind (131 lesson_withhold, 150 verify_toggle, 28 "
          "step_deny), and their effects are far smaller, so a PASS here "
          "does NOT license believing a `mattered_*` verdict on them.")
    print("  On anything but PASS, `dream_credit` stays closed and Dream "
          "produces reports, not labels.")


def build_context(home: str, upstream: str, with_vector: bool = True):
    """A real context, from the smoke driver — the two scripts must not
    build the environment differently, or a D4 verdict describes a
    configuration the nightly job never runs in."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from dream_replay_smoke import _build_context
    return _build_context(Path(home), upstream, with_vector=with_vector)


_STAMP = ".d4-owner.json"


def claim_scratch(scratch: Path, live: Path) -> str:
    """'' if `scratch` is safe to stage into, else why not.

    Two ways this went wrong before. (1) Both scripts' docstrings promise
    "runs against a COPY … nothing is written to the real home" and no
    code enforced it — `--home $GHOST_HOME` ran the whole thing against
    live state, and a `--home` sharing the live `system/memory` would put
    a second writer on the Chroma DB the live agent holds. (2) The
    default path is FIXED, so a second run `rsync --delete`s over one in
    flight; the victim does not crash, it degrades into abstains, i.e. it
    looks exactly like an engine that cannot detect anything.
    """
    s, l = scratch.resolve(), live.resolve()
    if s == l or l in s.parents or s in l.parents:
        return (f"refusing to run against the LIVE home: {s} overlaps {l}. "
                f"The live agent holds the Chroma DB.")
    stamp = scratch / _STAMP
    owner = None
    if stamp.exists():
        try:
            owner = json.loads(stamp.read_text())
            if not isinstance(owner, dict):
                raise ValueError(f"not an object: {owner!r}")
        except Exception as exc:                  # noqa: BLE001
            # ⚠ FAIL CLOSED ON AN UNREADABLE CLAIM. A stamp truncated
            # mid-write — which is exactly what a killed run leaves
            # behind — used to parse as "no claim at all", i.e. safe to
            # rsync --delete over. The benign input refused and the
            # dangerous one was waved through.
            return (f"{scratch} carries an UNREADABLE claim ({exc}) — a run "
                    f"may be in flight. Delete {stamp} if you are sure it "
                    f"is not.")
    if owner:
        try:
            pid = int(owner.get("pid") or 0)
        except (TypeError, ValueError):
            pid = 0
        if pid <= 0:
            # `os.kill(0, 0)` signals the CALLER's process group and
            # succeeds, so a stamp with a missing or zero pid read as
            # "claimed by a live run (pid 0)" forever and bricked the
            # directory.
            return (f"{scratch} carries a malformed claim "
                    f"({owner!r}) — delete {scratch / _STAMP} to reuse it")
        alive = False
        try:
            os.kill(pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False
        except PermissionError:
            # The PID exists and belongs to someone else. Treating a
            # permission error as "dead" is how a liveness check reports
            # every process it cannot signal as reapable.
            alive = True
        except Exception:
            alive = False
        alive = alive and pid != os.getpid()
        if alive:
            return (f"{scratch} is claimed by a live run (pid {pid}, started "
                    f"{owner.get('started')}). Staging over it would "
                    f"rsync --delete a measurement in flight. Use a "
                    f"different --home (it will be staged from scratch), "
                    f"or delete {scratch / _STAMP} if that pid is stale.")
    return ""


def stamp_scratch(scratch: Path) -> bool:
    """True when the claim was actually written.

    Swallowing the failure silently turned the protection OFF with no
    signal: an unwritable scratch dir meant no stamp, and the next run's
    `claim_scratch` then found nothing and happily rsync --delete'd over
    a measurement in flight.
    """
    try:
        scratch.mkdir(parents=True, exist_ok=True)
        (scratch / _STAMP).write_text(json.dumps(
            {"pid": os.getpid(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "argv": sys.argv[1:]}))
        return True
    except Exception as exc:                      # noqa: BLE001
        print(f"⚠ could not claim {scratch} ({exc}) — a concurrent run "
              f"could stage over this one", file=sys.stderr)
        return False


def report(out: dict, *, as_json: bool = False) -> int:
    """Score, print, and return the process exit code.

    Split out of `main` so the verdict → exit-code mapping is reachable
    from a test. It was previously a single line at the bottom of a
    function that needs a staged home, a live llama-server and docker to
    reach, so swapping two of the constants was invisible to the suite.
    """
    res = score(out)
    if as_json:
        print(json.dumps({"result": res, "detail": out}, indent=1))
    else:
        _render(out, res)
    return VERDICT_EXIT.get(res["verdict"], EXIT_MISS)


def build_parser() -> argparse.ArgumentParser:
    """Split out so a test can read the DEFAULTS the operator gets,
    rather than asserting on the source text that produces them."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--pairs", type=int, default=0,
                    help="legs per arm; 0 = the ENGINE's own default, which "
                         "is what a gate should certify")
    #: HEADROOM, not the floor. Defaulting to exactly MIN_STABILITY_N
    #: meant a single abstained null among the first N made `st_dec` one
    #: short and returned UNDERPOWERED for the whole run, unrecoverably,
    #: after hours of container work.
    ap.add_argument("--stability-cases", type=int,
                    default=STABILITY_ATTEMPTS)
    ap.add_argument("--leg-timeout", type=float, default=300.0)
    ap.add_argument("--max-hours", type=float, default=8.0,
                    help="wall-clock budget, checked BETWEEN cases — a run "
                         "can overrun by at most one case (3 spec budgets), "
                         "because clipping a case in flight corrupts it "
                         "rather than dropping it")
    ap.add_argument("--shuffle-seed", type=int, default=0)
    ap.add_argument("--upstream", default="http://127.0.0.1:8088")
    ap.add_argument("--home", default="",
                    help="scratch GHOST_HOME (default: a fresh copy of the "
                         "live one — the live agent holds the Chroma DB)")
    ap.add_argument("--no-vector", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rescore", default="",
                    help="score an EARLIER run's d4_rows.jsonl and exit — "
                         "the recovery path for a run that was killed")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    if args.rescore:
        if not Path(args.rescore).is_file():
            print(f"no such rows file: {args.rescore}", file=sys.stderr)
            return EXIT_NO_DATA
        return report(rescore(args.rescore), as_json=args.json)

    live = os.getenv("GHOST_HOME", "").strip()
    if not live or not Path(live).is_dir():
        print("GHOST_HOME is not set to a real directory — refusing to "
              "guess a relative path", file=sys.stderr)
        return EXIT_NO_DATA

    # Say it BEFORE spending hours, not after: the verdict of a run with
    # fewer cases than the power floor is decided before the first leg.
    if args.cases < MIN_CASES_PER_ARM:
        print(f"⚠ --cases {args.cases} < MIN_CASES_PER_ARM "
              f"({MIN_CASES_PER_ARM}): this run CANNOT return PASS. It is a "
              f"smoke run; the verdict will be UNDERPOWERED.", flush=True)
    if args.stability_cases < MIN_STABILITY_N:
        print(f"⚠ --stability-cases {args.stability_cases} < "
              f"MIN_STABILITY_N ({MIN_STABILITY_N}): this run CANNOT return "
              f"PASS.", flush=True)
    elif args.stability_cases < STABILITY_ATTEMPTS:
        # ⚠ MEASURED, not guessed. A five-hour run was launched with
        # `--stability-cases 10` — exactly the floor — and by case 9
        # three repeats had already abstained, so the maximum achievable
        # decided count was 7 and the verdict was UNDERPOWERED before the
        # remaining 19 cases ran.
        print(f"⚠ --stability-cases {args.stability_cases} leaves little "
              f"headroom: a repeat counts only when BOTH runs decide "
              f"(observed {STABILITY_DECIDE_RATE:.2f}), so reaching "
              f"{MIN_STABILITY_N} decided needs about "
              f"{STABILITY_ATTEMPTS} attempts.", flush=True)

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from dream_replay_smoke import _stage_home
    default_home = not args.home
    scratch = (Path(args.home) if args.home
               else Path(os.getenv("TMPDIR", "/tmp")) / "ghost-replay-d4")
    why = claim_scratch(scratch, Path(live))
    if why:
        print(f"REFUSING: {why}", file=sys.stderr)
        return EXIT_NO_DATA
    if not stamp_scratch(scratch):
        # Fail closed: an unclaimed multi-hour run is exactly the state
        # the claim exists to prevent, and one stderr line in an
        # unattended run is not a mitigation.
        print(f"REFUSING: {scratch} could not be claimed", file=sys.stderr)
        return EXIT_NO_DATA
    if default_home or not (scratch / "system" / "memory").is_dir():
        # ⚠ ALSO STAGE A NON-DEFAULT `--home`. Staging used to be
        # conditional on the DEFAULT path, so `--home /somewhere` ran
        # against an empty `system/memory`, `trajectories` and `sandbox`
        # — and the null arm ablates `recall`/`list_lessons`, which
        # against an empty memory are inert for a SECOND reason. The
        # specificity number would then describe a perturbation that
        # could not have had an effect either way. `claim_scratch`'s own
        # refusal message recommends "pass a different --home", so this
        # was one operator instruction away.
        print(f"staging a COPY of {live} → {scratch}", flush=True)
        _stage_home(Path(live), scratch)
        # NOTE: no re-stamp. `_stage_home` rsyncs `--delete` into SUBTREES
        # (`system/memory`, `system/trajectories`, `system/bench`,
        # `sandbox`) and the claim sits at the scratch ROOT, so it
        # survives staging — verified, not assumed. A re-stamp here would
        # be dead code implying a window that does not exist.
    # The seed challenges live under counterfactual/, which the smoke's
    # copy list does not carry.
    for tree in ("system/counterfactual",):
        src, dst = Path(live) / tree, scratch / tree
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dst, dirs_exist_ok=True)
    os.environ["GHOST_HOME"] = str(scratch)

    from ghost_agent.core.replay_engine import preflight
    ok, why = preflight()
    print(f"preflight: {'CLEAR' if ok else 'BLOCKED'} — {why}", flush=True)
    if not ok:
        return EXIT_NO_DATA

    rows_path = scratch / "d4_rows.jsonl"
    print(f"per-case rows → {rows_path}", flush=True)
    ctx = build_context(str(scratch), args.upstream,
                        with_vector=not args.no_vector)
    out = None
    try:
        out = asyncio.run(validate(
            ctx, cases=args.cases, n_pairs=(args.pairs or None),
            stability_cases=args.stability_cases,
            leg_timeout_s=args.leg_timeout,
            run_deadline=time.monotonic() + args.max_hours * 3600.0,
            shuffle_seed=args.shuffle_seed, rows_path=rows_path))
    except KeyboardInterrupt:
        print("\ninterrupted", flush=True)
    except Exception as exc:                      # noqa: BLE001
        # Anything escaping `validate` (a raising preflight, a corpus
        # read) used to propagate and exit 1 — which the documented table
        # reads as "a bar was missed".
        print(f"\nrun FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
    finally:
        try:
            (scratch / _STAMP).unlink()
        except Exception:
            pass
    if out is None:
        # Hours of container work must not vanish because the operator
        # hit ^C; the rows are on disk either way.
        print(f"no result rendered — partial rows are at {rows_path}",
              file=sys.stderr)
        return EXIT_INCOMPLETE
    return report(out, as_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
