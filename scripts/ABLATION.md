# Ablation harness — does the cognitive layer pay for itself?

The goal is one honest number per layer: **how much does turning it off change
task outcomes, and at what latency cost.** Anything that can't beat its cost
(complexity + latency + LLM calls) gets cut.

---

## Status & triage (2026-07-04)

The apparatus is **complete** (Track A auto+paired, Track B1+B2) but had
**never been run for real** — the only result was a 1-shot smoke of `full`. This
pass added the missing pieces and a cheap static triage that answers part of the
question without any sweep.

**New this pass:**
- `--no-verifier` flag + `full_no_verifier` config — the late verifier is now
  ablatable (it wasn't before). Tested in the strongest (inline) form; see the
  config note.
- `--suite behavioral` wired into `ablation_eval.py` — execution-grounded tasks
  (verified against the isolated agent's real sandbox / memory / DB), so a
  `tooluse` verdict can't be faked by plausible text. `--suite hard` (16 tasks)
  is also loadable now.

**Trace-to-consumer triage (production flags: no `--prm-model`, no
`--router-model`, no `--swarm-nodes`) — definitive WITHOUT a sweep:**

| Subsystem | Live consumer today? | Verdict |
|---|---|---|
| PRM (scoring/training) | **No** — `--prm-model` unset → MCTS/self-play/frontier read a neutral **0.5 placeholder** | INERT. MCTS's value-function branch runs on a placeholder. Wire a trained PRM (needs the discriminating suite as labels) or stop counting it. |
| Router / swarm dispatch | **No** — `--router-model` unset = "no-op that always allows full swarm"; no swarm nodes exist | INERT in production. |
| Reflection → lessons | **Yes** — injected as `### RELEVANT PRIOR LESSONS` (agent.py) | Live loop; magnitude = Track B. |
| Post-mortem *patch proposals* | **Human-only** — `proposal only — it is never applied` (main.py) | Cost (extra LLM calls generating diffs+repro on failed runs) for output only a human reads via the `postmortem` tool. Cut `--postmortem-propose-patch` if you don't read them. |
| Late verifier | Partial — next-turn correction is live; the PRM-feed is dead | Ablate: Track A = cost, Track B = the cross-session value. |

So two layers (PRM, router/swarm) are **provably inert now** — no experiment
needed; the decision is *wire them or delete them*. One (`postmortem-propose-patch`)
is pure cost unless a human consumes it. The rest need the sweep / Track B below.

This is deliberately built on the existing `ghost_agent.eval` harness
(`EvalSuite`, `CuratedRequestTask`, `aggregate`) — it adds the three things that
were missing: a **discriminating** task suite, **repeats + confidence intervals**
(the agent is stochastic), and a **config sweep + verdict** driver.

Files:
- `scripts/ablation_tasks.py` — the discriminating single-session suite. **Replace with your real workload.**
- `scripts/ablation_eval.py` — `run` (score the running agent N times) + `compare` (CIs + verdicts).
- `scripts/ablation_configs.json` — the boot recipe: FULL minus one layer, per config.

---

## The core method

1. **Fixed, auto-scorable tasks.** Objective validators, stable across runs. The
   seed suite has 13 across reasoning / tooluse / format / robustness / anchors.
   It is a starting point — the sooner it looks like *your* actual usage, the
   sooner the numbers mean something.
2. **Ablate one layer at a time** against a `full` reference. Each config is
   `full` minus a single layer (see `ablation_configs.json`).
3. **Repeat N≥5×** because temp>0 makes one run noise. Report success rate with
   a Wilson interval, latency with a mean CI.
4. **Pre-register the kill criterion** (below) *before* looking at results.
5. **Delete the losers.** A layer that doesn't move the metric beyond CI overlap
   — and isn't redeemed by Track B — comes out.

---

## Scoring — the `ANSWER:` line (and why "last number" was a trap)

The hard suite (`scripts/ablation_hard_tasks.py`) validates from the chat TEXT.
Every prompt is auto-suffixed with a directive to end on a canonical line:

    ANSWER: <bare value>

and the validators (`answer_int`, `answer_num`) read the **last `ANSWER:` line**,
not the last number in the reply. If the marker is missing they fall back to a
lenient "is the correct value present as a standalone token" check — so a
forgotten marker is not penalised, but trailing prose can never hijack the score.

**Why this matters — a real bug that corrupted a headline.** The old
`final_number_is` rule scored the *last numeric token* in the reply. A correct,
verified answer that shows its work ("the father is 36 … in 12 years 48 = 2×24 ✓")
ends on **24** and was marked WRONG. Because the verifier / deep-reason arms emit
more verification prose than `thin`, that rule docked them for *being verbose*,
manufacturing a ~3pp `full`-vs-`thin` gap that was pure formatting. The earn-keep
audit (2026-07-23) traced the entire "the full stack underperforms stripped"
finding to this: split by validator, the 12 lenient tasks were `full` 98% vs
`thin` 99% (a tie, and at a useless ceiling), while the 4 `final_number_is` tasks
were 39% vs 50% — the whole gap lived in the scoring artifact. **The verifier was
exonerated** (it CONFIRMED 100%; it never refuted a correct answer). Lesson: an
ablation's validator is part of its measurement — a validator that correlates
with an arm's *style* (verbosity) rather than its *correctness* will invent
differences that aren't there. Score the declared answer, not the last digit.

The trap tasks (`bat_ball`, `algae_quarter`, `avg_speed`, …) exist to add
failure-frontier headroom where reasoning/verification *can* pay off; their
parameters are shifted off the textbook values so a recalled trick doesn't win.

---

## Two tracks (this is the part people get wrong)

The single-shot suite only exercises **in-session** layers. Run cleanly (fresh
empty `GHOST_HOME` per config), the **cross-session** layers start blank and
*cannot* help on one-shot tasks. Don't read "no in-session effect" as "useless."

| Layer | Track | Probe |
|---|---|---|
| metacog / arbiter, deep-reason, pre-flight guard, swarm | **A** in-session | this suite |
| memory/RRF, selfhood, reflection, dream, skills_auto, cross-project map | **B** cross-session | retention protocol (below) |

### Track A — in-session ablation (this harness, today)

```bash
# stop the production agent first (avoid port/Docker clash), then per config:
TMP=$(mktemp -d)
GHOST_HOME="$TMP" GHOST_SANDBOX_DIR="$TMP/sandbox" \
  python -m src.ghost_agent.main --port 8010 --upstream-url http://127.0.0.1:8088 \
    --api-key '' --no-mandatory-tor <CONFIG FLAGS from ablation_configs.json> &
# wait for http://127.0.0.1:8010/api/version, then:
GHOST_HOME="$TMP" python scripts/ablation_eval.py run \
  --config <name> --base-url http://127.0.0.1:8010 --repeats 5
# kill the agent, repeat for the next config.

# when all configs are collected:
python scripts/ablation_eval.py compare --reference full
```

`compare` prints success rate + 95% CI + latency per config, the delta vs `full`,
and a verdict: **WORSE than ref** (removing it hurt → the layer earns its place),
**indistinguishable** (candidate to cut), **BETTER than ref** (the layer is
net-negative → cut). `--config matrix` prints the recipe.

### Track B — cross-session retention (the harder, more important one)

The cross-session layers claim "the agent gets better over time / across
sessions." To test that you need exposure → consolidation → fresh-session re-test:

1. Seed: run task A in session 1; let it fail or partially succeed.
2. Let the idle loops fire (reflection / dream / skill-mining / memory writes).
3. Re-test: present a **related** task in a **fresh** conversation.
4. Compare the session-2 score **with the layer on vs off** (`--no-reflection`,
   `--no-self-model`, `--no-memory`, …). If retention is identical with it off,
   the layer is theater.

This is **already built** (correcting an earlier note that said "next to build"):
- **`scripts/ablation_trackb.py`** (B1) — passive recall: seed→probe pairs,
  TREATMENT (memory ON) vs CONTROL (`--no-memory`), McNemar on matched probes.
- **`scripts/ablation_trackb2.py`** (B2) — cross-session RULE learning: task →
  correction → (consolidation delay) → probe, TREATMENT vs CONTROL, McNemar.
- **`scripts/ablation_trackb3.py`** (B3, 2026-07-07) — isolates the **pure-idle
  learning loops** (dream/self-play, idle reflection critique, skills-auto
  graduation) that B1/B2 could not reach because they only fire in long idle
  windows. It exploits two new flags (`core/agent.py` / `main.py`):
  - `--bio-time-scale N` divides every biological-watchdog idle window bound and
    phase cooldown by N (via `GhostAgent._bio_scaled`), so a 1-hour window fires
    after ~1 minute of idle. Default 1.0 = production timing. **Never set in prod.**
  - `--bio-deterministic` makes the probabilistic idle phases (dream 0.5,
    self-play 0.2) fire every eligible tick (`GhostAgent._bio_roll`) so the arms
    exercise the same phases each accelerated epoch.
  TREATMENT boots accelerated + deterministic; CONTROL boots at scale 1 (idle
  loops never reach their windows in the short run) — isolating the IDLE loops,
  not memory (Track B already validated memory). The harness compares probe
  outcomes (McNemar) AND the learning artifacts each arm produced during idle
  (playbook lessons by `source`, graduated-skill count, proposed macros). Covered
  by `tests/test_bio_time_scale.py`.
- Task banks: `trackb_tasks.py`, `trackb2_tasks.py`. `claude_trainer.py` was the seed.
All three are runnable today. **B3 FIRST LIVE PASS RUN 2026-07-07** (prod stopped
to free RAM; isolated arms against the shared llama-server): treatment (idle loops
accelerated, `--bio-time-scale 30 --bio-deterministic`) produced **1 `self_play`
lesson** ("alternative idiom for sql tasks", challenge SOLVED, score +1.900);
control (scale 1, loops never fire) produced **0**. First live proof a pure-idle
loop is productive (not dead code) — the thing Track B could not reach. Caveats:
1 repeat; dream did not fire (too few seed memories for entropy) and reflection had
0 failed turns; the probe-outcome McNemar (does idle output *improve* task success?)
needs richer seeds + more repeats. The harness works end-to-end (a report-builder
bug was fixed post-run — `_b3_report`).

---

## The kill criterion — fill this in BEFORE you run

> For each layer L, I will keep it only if removing it drops success rate by
> **≥ ___ percentage points** with the 95% CI of the difference **excluding 0**,
> OR (Track B) improves session-2 retention by the same bar. Otherwise I
> default it OFF and schedule its code for deletion. Latency: if an ablation is
> **indistinguishable on success but ≥ ___ s/task cheaper**, that is a cut.

Write the numbers down. Commit to them. The whole point is to not let every
layer survive on a single lucky run and a good story.

---

## Throughput reality

13 tasks × 5 repeats × 9 configs ≈ 585 agent calls. At ~30–60 s/call that's a
half-day to a day of unattended runs — fine overnight, painful interactively.
Triage: start with the **biggest, most expensive, most-doubted** layer
(deep-reason for cost; selfhood/dream for doubt) and run *one* clean ablation.
One honest answer beats an unfinished matrix.

A wide CI ("indistinguishable") on a small run means **underpowered, not "no
effect."** Add repeats or grow the suite until the CIs are tight enough to
decide — or accept that an effect too small to measure here is too small to
justify the complexity.

---

## Paired / time-matched mode (scripts/ablation_paired.py)

The `auto` sweep is SEQUENTIAL: each config is measured at a different moment,
so drift in shared-upstream load (e.g. a production agent on :8000 doing idle
work) confounds the comparison. `ablation_paired.py` removes that:

  * every config is booted at once, each on its own port + fresh GHOST_HOME
    (booted SEQUENTIALLY — boot, wait-ready, next — to avoid a bind race and
    halve peak boot RAM);
  * for each (repeat, task) the SAME task is fired at every config back-to-back
    in randomized order, within seconds — so whatever load the upstream is
    under at that instant is the same for all configs and cancels in the pair.

Statistics are PAIRED: McNemar's exact test on the matched pass/fail pairs
(only discordant pairs carry signal), plus a paired difference CI. The report
also prints **within-task variance across repeats**: if it is ≈0 the model is
effectively deterministic on the suite, so the effective sample size is the
number of TASKS, not trials — don't over-trust tight trial-based CIs; read the
per-task table as the real evidence.

Run (headline = total in-session lift of the whole stack):

    PYTHONPATH=src python scripts/ablation_paired.py \
      --configs full,thin --suite default --repeats 15 \
      --base-port 8030 --timeout 600 --report-dir <out>

Attribution (which layer): re-run pairwise against `full`, ONE ablation at a
time so only two throwaway agents share the upstream (keeps load — and the
deep-reason timeout bias — low):

    --configs full,full_no_deepreason   # then full_no_metacog, full_no_preflight

Caveat unchanged: fresh GHOST_HOME ⇒ cross-session layers (memory/selfhood/
workspace/reflection) cannot help single-shot; judge those with Track B.

## Track B4 — the grounded outcome battery (2026-07-09, journal §4D)

Track B3 answered "do the idle loops PRODUCE anything?" (yes — lessons) but its
fact-recall probes saturate at ~97% in every arm (memory is ON in both), so the
outcome question stayed open. B4 replaces the probes with execution-grounded
DOING tasks (`scripts/trackb4_tasks.py`): the harness writes deterministic
fixture files into the arm's sandbox, the prompt drives the live agent to
compute something and write a result artifact, and a pure-Python reference
implementation computes the expected answer from the SAME fixtures — the
verifier compares tokens (no LLM judge, no recall bottleneck). Every task is
gated self-consistent by the test suite (`tests/test_trackb4_battery.py`), the
same philosophy as self-play's reference-solution gate.

What's new vs the B3 driver (`scripts/ablation_trackb4.py`):

  * **Seeding phase (identical in all arms)** before the idle window — easy
    tasks feed strong clusters, hard tasks produce real failures in the
    pre-registered WEAK_CLUSTERS (sql, regex_parse, algo, concurrency):
    reflection material + frontier variance.
  * **`--smart-memory 0.9` in every arm.** B3's arms never passed the flag,
    and the smart-memory consolidator is the only writer of the `type:"auto"`
    fragments dream's ≥3-auto-memories entropy gate counts — dream was
    unsatisfiable by construction. Gate state is instrumented from the arm log
    (`Auto Memory Store` / `Not enough entropy to dream` line counts).
  * **Mediation capture** — per probe, did any playbook lesson actually
    surface? (retrieval-credit counters diffed around each request). An
    outcomes-null is uninterpretable without this: mediation≈0 → fix retrieval
    routing; mediation healthy → idle output doesn't transfer at this scale.
  * **Task-stratified sign-flip test** next to the exact McNemar — repeats
    within a task are correlated; (task,repeat)-independence over-narrows.
  * **`--pilot` calibration**: one control-configured agent, `--pilot-repeats`
    (default 3) passes over all candidates; keeps tasks that are neither
    all-pass nor all-fail and emits `b4_battery.json` (the implementable form
    of the [0.3, 0.7] difficulty band).
  * Fixture seeds vary per repeat (`DEFAULT_SEED + rep`) so a memorised answer
    can't carry across repeats, but stay identical across arms within a repeat
    (matched pairs need matched fixtures).
  * **Timeout-bleed guard (post-pilot 2026-07-09):** a client-side probe
    timeout does NOT stop the agent's in-flight turn; the next request queues
    behind the turn-serialization semaphore and burns its own budget waiting
    (pilot #1: one genuine 300s overrun took the next two tasks to 0/3 without
    ever measuring them). The driver now calls `_wait_arm_quiet` between every
    task — poll the arm log's `Request Finished` count vs requests sent,
    bounded grace — and every task has globally-unique fixture filenames
    (same-name/different-content fixtures got swapped under the still-running
    task). `--battery-file` also filters `--pilot` candidates so a re-pilot
    measures only new/re-raced tasks.
  * **Calibration lesson (pilot #1):** clean single-file shapes are CEILING
    for this model (all sql/algo/pg 3/3 at 42-91s); the in-band lever is
    messy MULTI-FILE data + fiddly-but-precise rules (all 4 data_analysis
    tasks landed 2/3). The v2 candidates port that recipe into the ceiling
    clusters.

Run (operator, prod stopped — journal §2 supervisor gotcha applies):

    PYTHONPATH=src python scripts/ablation_trackb4.py --pilot \
        --report-dir ablation_out/b4-pilot            # ~1.5-2 h
    PYTHONPATH=src python scripts/ablation_trackb4.py --repeats 3 \
        --battery-file ablation_out/b4-pilot/b4_battery.json \
        --report-dir ablation_out/b4-$(date +%Y%m%d)  # ~11-12 h overnight

Pre-registered verdicts (journal §4D items 5-7): idle-loop outcome value =
stratified p<0.05 with healthy mediation; frontier KEEP iff self-play yield ≥
uniform in ≥2/3 repeats AND weak-cluster probe delta ≥ 0, else flip default to
uniform (PRM stays either way); dream gate satisfied-but-silent = new bug.

## Earn-your-keep — the STANDING self-measuring / self-pruning harness (2026-07-22)

> **⛔ CLOSED as INCONCLUSIVE-for-this-model (operator decision 2026-07-23).** The synthetic-ablation
> route below is **dormant** — do not run more batteries expecting a prune verdict. The premise (auto-graded
> deterministic tasks discriminate whether a subsystem helps) fails on this uncontended 35B: both the Track-A
> puzzle battery and the Track-B4 grounded battery ceiling, so the auto-prune rule never fires. **Nothing was
> pruned; prod config is unchanged.** The code stays as reference/dormant infra; it would only be revived if
> the *instrument* changes (observational mediation on live trajectories, or a deliberately degraded model).
> See `PROJECT_JOURNAL.md` §6 "2026-07-23 (later 3)". The mechanics below are retained for that possibility.

`scripts/earn_keep.py` turns the one-shot ablations above into a standing
capability: every cognitive subsystem must prove it earns its keep or get
auto-pruned. It's a thin orchestration layer over the paired driver — no new
measurement machinery.

**What it does.** `run` boots the **leave-one-out** config matrix
(`ablation_paired.CONFIG_FLAGS`: `full` + one `full_no_<x>` per subsystem +
`thin`) and fires the hard battery at every arm back-to-back (paired → shared
upstream load cancels). The marginal contribution of subsystem X = the paired
`full` vs `full_no_x` delta. Every run's raw per-(arm, repeat, task) pass/fail
is APPENDED to a durable ledger (`ablation_out/earn_keep/results.jsonl`), so the
picture TRENDS across on-demand runs. `report` aggregates the ledger into a
ranked keep/prune table (Δ-help, 90% bootstrap CI, latency cost, pair/run
counts, verdict).

> **Ledger hygiene.** `report` pools EVERY run in `results.jsonl`, so the ledger
> may only contain runs of the *same* battery + scoring contract. When the tasks
> or validators change, archive the old ledger before re-running, e.g.
> `mv results.jsonl results.pre-<change>.jsonl` — pooling incompatible runs
> re-corrupts the report. (Done once already: `results.pre-answerline.jsonl`
> holds the 3 runs scored under the biased last-number rule; see the scoring
> section above.)

    # prod stopped (paired needs 2+ throwaway agents + the 35B):
    PYTHONPATH=src GHOST_HOME=<live-home> python scripts/earn_keep.py run --track A --repeats 3
    PYTHONPATH=src GHOST_HOME=<live-home> python scripts/earn_keep.py run --resume       # after a crash/kill
    PYTHONPATH=src GHOST_HOME=<live-home> python scripts/earn_keep.py report              # trend so far
    PYTHONPATH=src GHOST_HOME=<live-home> python scripts/earn_keep.py report --apply --dry-run

**Single script, resumable, monitorable.** `run` is the one entrypoint. Each run
gets a directory `ablation_out/earn_keep/run-<id>/` with three artifacts:

  * `progress.log` — the whole-process narrative (booting, per-group / per-cell
    PASS/fail, checkpoints, completion), teed to stdout AND the file, and it also
    absorbs the boot/teardown chatter — so `tail -f progress.log` mirrors exactly
    what the operator sees.
  * `<arm>.log` — each throwaway agent's own stdout/stderr (`full.log`, `thin.log`,
    `full_no_deepreason.log`, …) for drilling into one arm's behaviour.
  * `checkpoint.jsonl` + `manifest.json` — the resume substrate.

**Idle loops are quiesced for Track A** (`--bio-time-scale 0.001`, applied
uniformly to every arm). Track A measures per-turn cognition, which fires on the
task turns — but the ~9 arms share one llama slot (`-np 1`), and each arm's own
biological idle loops (dream / REM / self-play / reflection / autoadvance) would
otherwise fire between measured tasks and contend on the single GPU. That load
is per-arm *asymmetric*, which the paired design does NOT cancel, so it would
poison the latency/outcome signal and crawl the run. Quiescing pushes every
idle-window bound out to ~days so no idle phase fires; uniform across arms → no
bias. (Track B, which *measures* the idle loops, does the opposite.)

The run prints the exact `tail -f` commands for both at startup. Progress is
checkpointed at **(repeat, task)-GROUP** granularity — all arms for a task fire
back-to-back (so the paired-within-seconds property survives a resume) and a
group is written atomically, so a kill mid-group redoes only that one group.
`run --resume` (or `--run-id <id>`) picks up the latest incomplete run, skips
done groups, and continues; the run's data is folded into the durable
`results.jsonl` ledger ONLY when all groups complete (no partial run pollutes the
trend). Starting a fresh run while one is incomplete is refused (use `--resume`
or `--force-new`).

**Auto-prune (pre-registered rule — do not move post-hoc).** A subsystem is
auto-pruned only when, across **≥3 runs** totalling **≥60 matched pairs**, its
Δ vs `full` is **≤ 0** AND the 90% CI upper bound is **< +2 pp** AND it carries a
compute cost (`costs=True` in the catalog — a free-but-useless subsystem is
harmless to keep). **Protected set — measured but NEVER auto-pruned:** `memory`
(Track-B-proven, 98% vs 0%) and `verifier` (correctness guard).

**How a prune reaches prod (reversible).** The harness writes
`$GHOST_HOME/system/earn_keep/pruned.json`. At boot `main.py` reads it via
`core.prune_overrides` and flips the subsystem off — env-kind toggles (e.g.
`GHOST_HYPOTHESIS_GROUNDING`) BEFORE `core.agent` is imported, arg-kind toggles
after `parse_args()` — logging each loudly (`⚖️ Earn-your-keep …`). **Revert** by
deleting the entry (or the file); the subsystem returns on the next restart. The
catalog `core.prune_overrides.SUBSYSTEMS` is the single source of truth for the
subsystem↔arm↔toggle mapping.

**Phase 2 (needs an operator evening):** wire the Track-B idle-loop LOO
(dream/self-play/reflection arms, via the trackb4 protocol + a calibrated
battery) into `run --track B`, feeding the same ledger/report. That adjudicates
the idle loops — the one genuinely open question. Tests:
`tests/test_earn_keep.py` (attribution math, the verdict rule's every branch,
prune I/O + prod-apply + protected refusal — all pure, no live boot).

*Phase 2 progress (2026-07-23).* The idle-loop **disable toggles** now exist and
are catalogued as Track-B, costed arms:

| loop | toggle | arm | notes |
|---|---|---|---|
| reflection | `--no-reflection` | `full_no_reflection` | pre-existing |
| dream (REM, phase 2) | `--no-dream` | `full_no_dream` | gate at `agent.py` phase 2 |
| self-play (phase 3) | `--no-self-play` | `full_no_selfplay` | ablates fresh self-play **and** counterfactual replay; **≠** `--no-frontier-selfplay` (that only changes cluster *selection*) |

Each disables *only* its own loop (isolation tested in `tests/test_biological_watchdog.py`).
**Still to do before running the LOO:** (1) calibrate the probe battery —
`ablation_trackb4.py --pilot` (the Track-A ceiling lesson applies: an
un-calibrated battery measures nothing); (2) confirm the collective
`treatment` vs `control` effect is non-null before spending arms on per-loop
attribution; (3) wire `run --track B` to boot these arms through the trackb4
seed→idle→probe protocol and fold probe outcomes into the ledger with
`track="B"`.
