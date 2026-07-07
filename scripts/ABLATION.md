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
