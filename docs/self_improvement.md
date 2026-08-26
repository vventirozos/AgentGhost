# Self-Improvement Pipeline

Ghost Agent's Stage 1 self-improvement substrate. All six modules run
fully local — no external teacher, no hosted embedder, no outbound
API. The reasoning chain: **measure → log → route → optimise →
acquire → reflect**, each step producing signal the next needs.

**Status:** wired end-to-end in `main.py`, verified against the live
agent on **two complementary timescales**:

* **real-time** (post-turn) — a user-correction-shaped follow-up
  promotes the prior trajectory to FAILED, schedules
  `Reflector.reflect_one` as an `asyncio.create_task`, and the
  composite sink writes the lesson to `SkillMemory` before the user
  types their next message. No idle window required.
* **idle backstop** (biological watchdog phase 2.5) — the original
  path. Trajectories the user never returned to correct (or whose
  correction was missed by the heuristic gate) get reflected on
  during the 15-60 min idle window.

Both paths share the composite sink and the `_reflected_trajectory_ids`
dedup set, so a trajectory reflected by either path is skipped by
the other. Verified live: a seeded failure produces a reflection
whose diagnosis + plan is persisted into `SkillMemory`, retrieved by
the memory bus on a fresh-but-similar user turn, and visibly changes
the agent's first action — all without any weight update.

### Auto-FAILED labelling — structured `ToolCall.error` on the chat path (2026-07-07)

The UNKNOWN→FAILED promotion for chat turns (`distill.outcome_heuristics`)
already existed with conservative flood control (repeated-selector,
≥3-identical-error, verifier-REFUTED, interact-abort signals; never demotes
PASSED/FAILED). This closes the deferred item #5 as **landed** by removing its
one residual weakness: the repeated-identical-error signal used to depend on
regex-sniffing result *text* because the chat recorder left `ToolCall.error`
empty (only self-play/batch set it), so failures with atypical text — e.g. the
native-tools tool-call-corruption shapes — were missed. `_record_turn_trajectory`
now populates `ToolCall.error` (a short normalized signature) whenever a paired
tool result looks like a failure, and the heuristic (`_tool_call_failed`) checks
the structured flag first with the text sniff as fallback for legacy
trajectories. Covered by `tests/test_trajectory_failure_heuristic.py`.

## Module map

| Module | Purpose | Wires into |
|---|---|---|
| `ghost_agent._env` | Telemetry hardening (single source of truth) | `main.py` import + `probe:telemetry_disabled` |
| `ghost_agent.eval` | Trust-aware eval harness (offline invariant gate + protective capability baseline + **execution-grounded behavioral suite**) + network egress guard | CLI `scripts/eval_baseline.py` (`gate` \| `freeze`/`compare --suite {default,capability,behavioral,offline,post_learning} --runner {stub,http}`; `behavioral` forces its own agent-driving+verifying runner) |
| `ghost_agent.distill` | Trajectory JSONL logs + N-sample self-consistency | `$GHOST_HOME/system/trajectories/` |
| `ghost_agent.router` | 18 hand-crafted features + a 384-d request embedding → numpy logistic regression → dispatch (§4BQ; `GHOST_ROUTER_EMBED=0` reverts to lexical-only) | `core/agent.py::handle_chat` (body\["_router_decision"\]) |
| `ghost_agent.optim` | DSPy/GEPA prompt optimisation (scope-gated) | Tunes planning / tool-selection / reflection prompts |
| `ghost_agent.skills_auto` | Passive skill extraction from validator-passing trajectories | Biological phase 2.6 (extract → consolidate → verify → graduate to `auto_skills.json`; PASSED input flows from the late-verdict backfill, 2026-07-05) |
| `ghost_agent.reflection` | Self-critique biological phase on FAILED trajectories | Biological phase 2.5 → composite sink (JSONL + SkillMemory) |
| `ghost_agent.reflection.postmortem` | Whole-transcript post-mortem → classified, durable defect reports (behavioural / configuration / code_defect) | Biological phase 2.5c (`--postmortem`) → SkillMemory (behavioural) + `DefectQueue` (`$GHOST_HOME/postmortem/`) → `postmortem` tool |
| `ghost_agent.prm` | Per-step value model — scores `(state, action)` for MCTS lookahead | Biological phase 2.7 (retrain, **consumer-gated** — skips unless `.score()` or `.uncertainty()` is live; §4BN) → `core.mcts.MCTSReasoner` (fast scoring) |
| `ghost_agent.selfhood` | First-person autobiographical diary + self-state + recognition / wake-up + narrative consolidation | Biological phase 2.8 (narrative regen) → wake-up prefix on every `handle_chat` |

## The flow (as wired)

```
       user request
            │
            ▼
     ┌──────────────┐
     │   router     │  decision stashed on body["_router_decision"]
     └──────────────┘   (fail-safe: escalates to full swarm when unsure)
            │
            ▼
       agent turn ─────────────────┐
            │                      │
            ▼                      ▼
  _record_turn_trajectory    response to user
  → TrajectoryCollector
    .append (redacts, day-
     partitioned JSONL)
  → stash (response_fp → traj)        ◄─ enables real-time path
    on ctx._recent_trajectories
            │
            ▼
   NEXT user turn arrives
            │
            ▼
   _maybe_promote_prior_turn_via_user_correction:
     (A) anchored correction-phrase regex
         on the new user message AND
     (B) Jaccard token-overlap of new
         user vs prior user ≥ 0.40
     ──── BOTH must fire ────
                │
                ▼
        update_outcome → corrections.jsonl  (overlay, audit-safe)
        traj.outcome   = FAILED              (in-memory)
        asyncio.create_task(
            Reflector.reflect_one(traj, sink))    ◄─ fire-and-forget,
                                                     real-time path
            │
            ▼
   ┌──────────────┐
   │ biological   │  idle-time phases (backstop):
   │   watchdog   │   1.   journal drain       (>120s)
   └──────────────┘   2.   REM dream           (600-3600s, cooldown 30m)
            │         2.5  reflection          (900-3600s, cooldown 40m)
            │         2.6  skills auto-extract (900-3600s, cooldown 2h)
            │         2.7  PRM retrain         (900-3600s, cooldown 3h)
            │         2.8  selfhood narrative  (900-3600s, cooldown 1h)
            │         3.   self-play           (>3600s,   cooldown 60m)
            ▼
  FAILED trajectories ─→ Reflector.run ─→ (diagnosis, plan)
                                          │
                  ─────── BOTH paths converge here ────────
                                          ▼
                         composite sink ──┬──→ JSONL (task_kind=reflection)
                                          └──→ SkillMemory.learn_lesson
                                               (retrieved by memory bus
                                                on next similar user turn)
```

> **`response_fp` is banner-insensitive.** The stash keys the trajectory (and the
> paired calibration components) by `_response_fingerprint` of the *response text*,
> but the next turn looks it up via `messages[-2]` — the message the client echoed
> back, which carries any banner this agent deterministically **prepends** to a
> reply: the async-verdict correction (`⚠️ **Correction to my previous answer:** …`),
> a clarifying-question lead-in, or an autonomous-progress digest — each terminated
> by a `\n\n---\n\n` separator and stacked in front of the body. A raw prefix hash
> shifted on the banner and missed, silently dropping the "confidently wrong"
> calibration negative + FAILED promotion on exactly the hedged/corrected turns that
> matter most. `_response_fingerprint` now peels leading banner blocks
> (`_strip_leading_banners`, bounded so a genuine long intro before a markdown rule
> is left intact) before hashing, so stash and lookup agree whether or not a banner
> is present. Covered by `tests/test_correction_fingerprint_banners.py`.

```
   on-demand:
            │
            ▼
     eval/suite.run → SuiteResult → diff vs baseline.json
       (--suite default        → regression + capability + curated + template)
       (--suite post_learning  → 5 file-read-shape prompts that score
                                 "discover before reading" behaviour)
```

## One outcome, one source of truth

Every turn produces a single quality verdict via
`distill.outcome_heuristics.resolve_turn_outcome(current, verifier,
execution_failed, current_reason, unacknowledged_total_failure)`, applied in
`_record_turn_trajectory` before the trajectory is appended — and *called*
(not re-stated) by the late-verdict backfill, so both delivery paths share one
ladder. Priority (strongest first): a **verifier**-REFUTED verdict (conf ≥ 0.7)
→ FAILED; an existing FAILED (shape heuristics) is never upgraded away, except
one stamped exactly `structural failure`, which a late CONFIRMED may lift
(2026-07-31 honest-failure rule); a verifier-SUPPORTED verdict → PASSED —
**unless** every tool call in the turn failed and the reply never said so
(2026-08-04 shape rule, `unacknowledged_total_failure`, kill switch
`GHOST_UNACKED_FAILURE_GATE=0`); a **structural** execution failure (non-zero
exit / tool error) → FAILED; otherwise UNKNOWN.

This matters because the outcome is what the whole spine keys off: the Reflector
only reflects on FAILED trajectories, and the PRM only trains on
PASSED/FAILED. Before consolidation the verifier verdict reached calibration and
the selfhood model but **not** the corpus, so a verifier-caught wrong answer
stayed UNKNOWN and silently never became a lesson or a training negative. The
`probe:outcome_consolidation` invariant in the offline gate locks the priority
order in place.

## Privacy guarantees (strict local-only)

* `ghost_agent._env` sets every telemetry opt-out at import time
  (`ANONYMIZED_TELEMETRY`, `POSTHOG_DISABLED`, `TELEMETRY_IMPL`,
  `CHROMA_TELEMETRY_IMPL`, `HF_HUB_DISABLE_TELEMETRY`,
  `DISABLE_VERSION_CHECK`). `check_disabled()` is the probe's verdict
  — adding a new required flag in `_REQUIRED_FLAGS` is picked up
  automatically by the regression probe.
* `eval/network_guard.no_external_network()` — opt-in context manager
  that raises `NetworkEgressError` on any non-loopback socket connect.
  `scripts/eval_baseline.py` wraps the whole suite run in it.
* `distill/redact.redact_trajectory` — runs inside every `collector.append`.
  Strips API keys (OpenAI, Anthropic, Slack, GitHub, AWS), bearer
  headers (HTTP and JSON-quoted), `.onion` addresses, emails,
  `/Users/<name>` / `/home/<name>` paths, and non-loopback IPv4.
  Idempotent and order-preserving. Verified end-to-end in the live
  test: a prompt containing `sk-liveABCDEFGH...` and `/Users/alice/...`
  landed on disk with `<REDACTED_API_KEY>` / `/Users/<user>`.
* `optim/run_gepa` uses **Ghost's own upstream** as the optimizer LM
  via the `_GhostLMAdapter` wrapper — no teacher endpoint anywhere.
  `dspy-ai>=3.2.0` is listed in `requirements.txt`; the wrapper
  defers `import dspy` to call sites via `_require_dspy()` so a
  broken install surfaces a clear error instead of a cryptic
  `ImportError` during module load.

### Optimizer eval hygiene (§4F Phase 0, 2026-07-29)

Any loop that rewrites its own prompts must not be allowed to grade
itself. Measured context: proxy-gaming rates in self-optimizing agents
run 46–74% and *rise* with optimization steps (26%→58% between 10 and
100); self-critique does not fix it, a hidden holdout does. Three
mechanisms enforce this, all live before the first real GEPA run:

* **PUBLIC/PRIVATE example split** — `trainset.split_public_private`
  assigns every example a tier by `sha256` of a stable identity
  (`source_trajectory_id` when present, content key otherwise; the
  identity deliberately excludes `signature_name` so one trajectory
  lands in the same tier for *every* signature). Membership therefore
  never migrates as the corpus grows — unlike a seeded positional
  shuffle, which re-deals membership each run and slowly leaks
  training examples into the "holdout". `scripts/run_gepa.py` gives
  the optimizer (train + its internal val split) ONLY the public tier;
  the A/B ship-gate (`ab_eval.compare_prompts`) judges ONLY the
  private tier (default `--private-pct 30`). An empty private tier
  refuses promotion rather than falling back to public examples.
  The public val split is forwarded to the optimizer as its own
  candidate-selection set (`run_gepa(..., valset=...)`, with a
  TypeError fallback for tuners that don't accept one) — GEPA picks
  its Pareto frontier on the valset, which is exactly why that set
  must be public-tier, never the ship-gate holdout.
* **`MAX_OPT_ITERATIONS` (=16)** — clamped inside `run_gepa()` itself
  so every caller inherits the cap; raising it is only legitimate
  alongside a stronger private holdout.
* **Activation telemetry** — `optim/loader.py` counts tuned-vs-baseline
  applications per signature (`activation_stats()`), and
  `learning_health` pairs artifacts-on-disk with those counters
  (`PROMPT OPTIMIZATION` section; `introspect action='learning'`).
  A tuned artifact with zero applies since boot renders with a ⚠ flag —
  this exact loop was once write-only, and the field's 2026 finding is
  that harness components failing to *fire* (not bad content) is the
  dominant failure mode. Counters survive `clear_cache()` by design.
  Operational rule: **never call `clear_cache()` on a live process** —
  a mid-session prompt reload shifts the KV stable-prefix pin; retrain
  offline, deploy via restart.

Tests: `tests/test_optim_eval_hygiene.py` (split stability under corpus
growth, cross-signature tier consistency, clamp-before-optimize, counter
semantics, learning-health pairing + ⚠ render).

### Verifier prompt optimization (§4F Phase 2, 2026-07-29)

The two-stage verifier templates (`_VERIFY_ENUMERATE_PROMPT`,
`_VERIFY_ADJUDICATE_PROMPT`) are GEPA-optimizable text assets:

* **Read-site** — `verifier._stage_template(name, baseline)` resolves
  in-process override (offline optimizer hook `_TEMPLATE_OVERRIDES`) →
  GEPA artifact via `optim.loader` (which also feeds the activation
  counters) → baseline constant. A tuned template is accepted ONLY if a
  **probe-format with dummy values succeeds AND every placeholder is
  present** (`_validate_stage_template`) — this rejects candidates that
  lost `{claim}`-class placeholders or broke the `{{ }}` JSON-brace
  escaping, which would otherwise raise inside `verify_claim` at
  runtime. Rejection logs a warning and falls back to the baseline.
* **Optimizer** — `scripts/optimize_verifier.py` uses the standalone
  `gepa` library with a custom adapter over the REAL pipeline: each
  candidate (both templates, two components) is evaluated by running
  fault-injected bench trials (`eval.verify_bench`) through
  `Verifier.verify_claim` against the judge endpoint that serves VERIFY
  in production (`--base-url`, the worker node). Scores are graded
  verdict-correctness; reflective feedback names the injected fault the
  verdict missed. A fresh client is built per evaluation loop (httpx
  pools are loop-affine; the adapter runs one `asyncio.run` per
  candidate).
* **Which pipeline the gate measures — `--escalate` (2026-08-04).**
  Production verifies with two models: a cheap judge, then a MAIN-model
  re-adjudication of every `REFUTED` **and** of every high-stakes
  `CONFIRMED`. A single-endpoint judge client cannot escalate, so the
  default `--escalate off` gates on the **cheap judge standalone** — and
  on the live recorded corpus (2026-07-30..08-04) the main model
  overturned **42 of 50 (84%)** of that judge's refutes, i.e. most of the
  false-alarm mass a candidate could be credited for is mass production
  already removes. `--escalate gate` (with `--main-base-url`) runs the
  *ship-gate* private evaluations through `EscalatingChatClient`, making
  the promotion decision production-equivalent while GEPA still trains
  against the cheap judge; `--escalate all` escalates everywhere (much
  slower). The CLI refuses `gate`/`all` without a main URL rather than
  silently running the raw arm, and every promoted artifact records
  `gate_arm` / `train_arm` / `gate_judge` / `gate_main`, so two artifacts
  judged by different pipelines can never be read as one series of
  `private_candidate_balanced` numbers.
  > ⚠ **The CONFIRM direction cannot move this gate's metric.**
  > `_trial_score` is verdict-only and `_escalate_confirm` never changes a
  > verdict — it caps confidence. So under `--escalate gate` the confirm
  > direction costs one main-model call per high-stakes CONFIRMED private
  > trial and contributes nothing to `balanced_score`. It is kept anyway,
  > because a gate that measures a *different* pipeline from production is
  > the defect being closed; making the score actionable-confidence aware
  > is a deliberate design change, not something to slip in silently.
* **Recording the baseline — `--incumbent-only <path>`.** Evaluates the
  LIVE incumbent templates on the private tier and exits without running
  GEPA, writing the balanced score with full provenance: pool sha256,
  fault set, template hashes, class mix, smallest resolvable delta, the
  escalation arm, both endpoints, the observed escalation events
  (refutes overturned / confirms eligible / confirms withheld), the
  cheap-leg `route_health`, and one row per trial so the number is
  re-scorable without re-running. That file is the number the next
  round's ship gate compares against — re-record it whenever the pool,
  the templates, the judge or the arm changes, because a baseline
  without those is not comparable to anything.
  > ⚠ **A cheap-leg timeout is not a verdict.** A failed `route()` call
  > falls through to the MAIN model, so that trial was judged by the
  > strong model, not the judge under test. `route_health.
  > fell_through_to_main > 0` means the baseline is a BLEND; the script
  > says so on stderr rather than letting it pass as a clean number.
  Tests: `tests/test_optimize_verifier_arm.py`.
* **Hygiene** — bench CASES hash-split public/private via
  `holdout_tier("vbcase:<id>")`; the optimizer trains on public trials
  only; the ship-gate compares baseline vs candidate on PRIVATE trials
  and promotes both artifacts (`verifier.enumerate.json`,
  `verifier.adjudicate.json`) only on `delta > --min-delta` AND a final
  placeholder re-validation; rejects persist as `*.candidate.rejected`.
  Iterations clamp to `MAX_OPT_ITERATIONS`.
* Run `scripts/verify_bench.py` against the same judge BEFORE
  optimizing (baseline TPR/FPR) — the standing rule for any verifier
  change, including judge-model swaps. Run it in the **same arm** you
  intend to gate in: a `raw_judge` baseline and a `judge+escalation`
  post-measurement are two different systems, and since 2026-08-04 the
  report names the arm and refuses to emit an unqualified `fpr` key.

Tests: `tests/test_verifier_tuned_templates.py` (baseline self-probe
regression, placeholder/brace rejection, override/artifact resolution
order, activation counting, formats-cleanly end check).

### verify_bench case pool (2026-08-04)

`verify_bench` is the one controlled instrument the §4F verifier work is
judged on, so its case pool has to represent production. Four things were
wrong and are now fixed:

| | before | after |
|---|---|---|
| harvest | classic prompt only — the two-stage **fallback**, 55 of 618 calls | + enumerate (281 calls) |
| poisoning | turns production REFUTED entered as clean `CONFIRMED` cases | verdict read from the same day-file; **62 of 106 (58.5%) excluded** |
| determinism | one shared RNG — growing the pool rewrote 20% of existing trials | seeded per (case, fault); only `wrong_topic` can move |
| provenance | a file **path** in `results.json` | `cases_sha256`, `n_cases`, `GHOST_HOME`, live template hashes |

The inversion needed one non-obvious thing: the templates carry `{{`/`}}` for
their JSON output spec, which `.format()` collapses, so matching raw template
segments against a rendered prompt fails ~800 characters into the tail. Before
the unescape, **0 of 580** live verify records inverted while their opening
sentence matched perfectly.

**Why it mattered:** the private tier was 4 cases / 30 trials, whose smallest
resolvable delta is `0.5/6 = 0.0833` — more than **four times** the
`--min-delta` of 0.02 it was being compared against. That is the arithmetic
behind the journal's "±0.08 private-gate noise", and it means the +0.087 ship
of 2026-07-30 was roughly one flipped trial. With the mined pool the private
tier is **29 cases / 220 trials** and the step is **0.0093**, finer than the
gate. **All three optimizer runners** — `optimize_verifier.py`,
`scripts/run_gepa.py` and (since 2026-08-04) `optimize_tool_descriptions.py` —
now REFUSE to run when a private tier cannot resolve its own threshold, and
`optimize_verifier.py` loads the mined pool **by default**, because without it
the gate resolved to 0.0833 and refused to run at its own default flags.

### The second of three ship gates (§4DA, 2026-08-25)

§4CY fixed `optim/ab_eval.py` and left two siblings carrying the rule it
replaced. `scripts/optimize_tool_descriptions.py` read

```python
ships = valid and aggregate_ok and delta > args.min_delta
```

— a margin with no significance test, which under the null promotes 25-40% of
the time because the smallest swing clearing the bar is one or two flipped
replays. It now requires the margin **and** McNemar `p <= ab_eval.SHIP_ALPHA`,
one-sided toward the candidate, reading the same constant as the GEPA gate and
the §4CW seed veto. `--allow-insignificant-ship` is the same recorded override,
and the pre-flight gained the significance floor beside its resolution check so
an unwinnable run costs nothing rather than `iterations × len(pub)` main-model
calls.

**McNemar is right here and wrong in §4CZ, and the difference is the point.**
Both arms replay the *same* fixture list in order, so fixture *i* is a matched
pair and a sign test applies. §4CZ's live arms are different requests —
unpaired — and need Fisher. Reaching for whichever test is to hand produces a
number for the wrong question. The gate refuses outright rather than pairing by
position across trajectory lists of different lengths.

> ⚠ **My stated reason for doing this gate first was wrong.** I wrote "192 mined
> fixtures against a threshold of 200". The pool holds 192 **rows** but only
> **66 positives**, and the supply gate counts positives — the real gap is 66 of
> 200, further away than GEPA's, not nearer. That is precisely the
> rows-vs-positives confusion this script's own docstring and
> `test_the_supply_gate_counts_positives_not_rows` exist to prevent, and I
> walked into it while citing the script. Its 13 private positives also give a
> resolution step of 1/13 = 0.077 against a 0.02 bar, so a real run refuses at
> the pre-flight today.
>
> The port is still worth having — this gate carried the defect §4CY spent seven
> rounds characterising, and fixing it costs a day. But it is not "the gate most
> likely to fire", and nothing here is running.

**Round 2 changed what this gate decides on, and what can read the result.**
The first version excluded transport failures from the *pairing* and left the
*margin* computed over every row, where a transport failure scores 0.0 — so
`--allow-insignificant-ship`, which ships on the margin alone, still promoted a
6-replay one-arm outage as a +0.100 win on identical descriptions. The margin is
now derived from usable pairs inside `_ship_decision`, and `delta` is no longer a
parameter: a caller could hand it a number the trajectories do not support, and
both callers did. An outage that walks the usable tier below the number the
pre-flight required to *start* now blocks the ship outright — the override waives
significance, never evidence.

The promoted artifact also had to change shape. It stamped its seven evidence
fields flat and called the count `discordant_replays`; `run_gepa.py` nests the
same seven under `gate` as `discordant_pairs`, and `recheck_gepa_incumbent.py`
reads `art["gate"]`, so the audit trail added so an override could be re-examined
was unreadable by the only instrument that reads such trails. It also omitted
`gate_arm`, the key `optim/loader.py` uses to decide whether an artifact has
provenance at all — so production logged a promotion made *under the current gate*
as *"predates the gate schema"*. One vocabulary now, plus `promoted_utc`, plus
both margins (raw and paired: they differ exactly when transport failed), plus a
`.prev` backup taken before the live artifact is overwritten and a promotion that
aborts if that backup fails.

One thing the round-2 fix itself broke, worth stating because it is the
recurring shape: marking a transport failure created a *new* error state, and
`make_reflective_dataset` was still reading the old two-valued world — it skipped
`"unreplayable"` and treated every other error as a per-tool-cap rejection, whose
feedback tells the reflector its description is too long. A llama-server restart
mid-run — the exact event the marker exists for — therefore taught the optimizer,
on every affected fixture at once, to shorten a description that was fine. Both
markers now come from one list.

Two numbers were simply wrong and are now real-over-real: the refusal's "collect
~N more positives" divided by *all* positives including bench (608 against the
miner's 181 on the same mine), and the **supply gate counted bench too** — the
hazard named in its own neighbouring comment, which §4BF-1c had closed for the
tier and not for the gate. A fresh mine is 403 positives of which 121 are real.

**Round 11 found the un-stamping was one-armed.** The read site prunes a
turn's attribution stamp when it refuses the artifact — but a control turn is
served the baseline and returns *before* the validator, so control stamps were
created and never pruned while treatment's were pruned on every refusal. Since
the aggregate ceiling is a property of the turn's tool subset, the attrition was
turn-shaped and one-sided: over 200 turns with an artifact neutral **by
construction**, `KEEP p=0.8020` became `REVERT p=0.0001`. Both refusal points now
ask what the withheld arm *would* have rendered and prune symmetrically.

**§4DC — the first autonomous GEPA actors (Phase 0+1).** The goal is a
fully autonomous mine→optimize→gate→promote→judge→revert loop; the §4DA
hardening was the prerequisite, because an autonomous caller acts on exit
codes. Phase 0 (supply watch) re-mines the fixture pool weekly and notifies
once when the miner's gate flips parked→ready. Phase 1 (live judge) runs
`gepa_live_check --revert` daily over every live artifact and acts only on
the declared contract: KEEP/could-not-measure are log-only, a REVERT retires
the artifact (the one autonomous action — it can only undo GEPA's own work)
and notifies that a restart is still the operator's, an undeclared code is an
instrument failure that acts on nothing. Cadence is wall-clock and persisted
(`system/gepa_autonomy_state.json`), notifications fire on transitions
with re-arming, and two kill switches exist: `GHOST_GEPA_AUTONOMY=0` (master)
and `GHOST_GEPA_AUTO_REVERT=0` (judge becomes report-only). Phases 2-4
(loader hot-reload to remove the root-restart dependency, autonomous
optimizer runs, closing the loop) are journaled in §4DC.

**Post-redesign round 2 closed §4DA.** The final pass mutation-tested round
1's own diff and found `run_gepa` promoting BEFORE validating (a schema
refusal could leave the candidate live and unstamped — the stamp is built and
validated before `os.replace` now, and the pin checks the disk), a helper
`SystemExit(<string>)` making the default invocation exit 1 ("a measured
rejection") for a missing fixture pool, and four pins that could not fail —
including the `disabled=` binding at its only production call site, where
`disabled=None` had survived 1,325 tests because the post-filters keep the
prompt correct and only the attribution regresses. All fixed, 7/7 mutants
killed. `optimize_verifier.py` remains outside the contract by operator
decision, behind a pinned perimeter test.

**Post-redesign round 1: the names held, the bindings did not.** Three
lenses attacked the redesign. No defect used an unregistered key or an
undeclared code — that class is closed — but the seed-arm schema's first
writer swapped the two rate fields (the artifact promoted *because it lost*
recorded a win; the schema now enforces `delta == seed_rate −
candidate_rate`, the one check a swap cannot pass), an outage recorded a veto
that never ran (`undecidable` is its own field now), two judge codes still
inverted on noise, `disabled_tools` was filtered after serving (exposure-free
treatment turns would drag a true REVERT toward KEEP), and a per-query cache
let one request serve twice with the last call owning the stamps. The guard
layer itself was unexecuted — `validate_gate_record` had zero calls in the
test tree and its refusal was swallowed in one gate — so conformance now
validates the artifact ON DISK through both gates, the reader scan fails
closed on unknown names, and `main()` may return only literal-valued
expressions.

**After sixteen rounds the review process itself was the problem.** The
ship rule converged at round 3; almost every later finding was one of two
shapes. Either a concept with no single definition, restated in each file
that touches it (17 files hard-code the arm labels, 7 the gate-record keys,
four instruments return 40 raw exit codes) — a review round samples one
pair of restatements, and there are quadratically many. Or a hand-picked
fixture family with the defect living exactly where it was thin, so each
round widened it by one axis and found the one mutant that axis exposed.
Round 16's own fix reproduced round 7's defect verbatim, which settled it.
The redesign gives the vocabulary one home (`optim/gate_contract.py`, with
an AST conformance test that fails on ANY divergence rather than the one a
reviewer sampled) and replaces the picked fixtures with an enumerated
world-space asserting invariants (`tests/test_read_site_invariants.py`),
whose ceilings are derived per world rather than chosen — the first
version used constants and let two known defect classes through.

**Round 16 found the turn's attribution describing a tool set the model
never saw.** Applying the tuned descriptions draws an arm, stamps the request
and prunes that stamp — all properties of the tool list passed in — so a second
call with a different list overwrites the first. The planner's *name list* used
the un-routed superset while the prompt build (the routed subset) was cached per
request, so from turn 2 the name list was the only call: the turn rendered the
artifact and kept no stamp, which blinds the live check and mines the turn into
the pool that ship-gates the next run. Round 14's `excluded` bucket was likewise
left outside the era filter, so turns that busted a *retired* artifact's ceiling
were reported as this one busting it. Exit 3 could not fire for the case it
documents (a byte-identical candidate scores a delta of exactly 0 at temperature
0, so the branch's `ships` guard was never true). The gate also **ratcheted**:
it seeds from the live artifact, so run N compares artifact-(N−1) with
artifact-N and the hand-written text is in neither — §4CW's seed veto is ported,
paid only on a re-promotion whose main gate already passed. And `recheck`
required significance to say KEEP but none at all to say RETIRE, on the same
evidence in both directions.

**Round 15 found the RETIRE exit code pinned by its own source text.** The
fourth code round 13 added — "this holdout cannot settle the question", as
distinct from "still wins" — was guarded only by an assertion that the characters
`return 2` appear near the branch. Inserting `_unmeasurable = _unmeasurable and
False` restores the pre-fix behaviour and survives the entire battery; the ten
tests that actually reach the branch asserted `rc in (0, 1, 2)`, which admits it.
Every exit code across the instruments is driven now. There are TWO exit
contracts, deliberately — a claim of one shared contract stood here and was
false: a GATE's `0` means the incumbent was REPLACED, a JUDGE's `0` means it
STANDS. Gates: 0 promoted, 1 rejected, 2 could not measure, 3 no candidate.
Judges: 0 still earns its place, 1 it does not, 2 could not measure, 3
reported but not acted on. What all four share is `2` — `gepa_live_check.py`
had been returning `0` for KEEP, INSUFFICIENT, CONFOUNDED and an unactioned
REVERT alike. Round 13's
`best -> _changed` fix also covered only the promote path, so a *rejected* run
still wrote records byte-identical to the incumbent carrying the set's statistic;
and `gate_scope` disclaimed a per-component measurement in exactly the case (one
changed component) where the A/B had made one.

**Round 14 found that fix arm-dependent again, and deleting a stamp it needed.**
The symmetry held only while every artifact was at least as long as its baseline
— a *shorter* one made a control draw bust the ceiling where a treatment draw did
not, giving `REVERT p=5.16e-06` on a neutral artifact. The prune now reads an
arm-invariant worst case. And the branch that fires when this arm's own render
was fine deleted the stamp anyway, which not only blinded the live check but
un-flagged the turn for the fixture miner — the same stamp answers "compare this
turn" and "this turn's context was mutated", and only the first had been
considered. Those turns are marked `excluded` instead.

**Round 13 found the gate promoting components it had never measured.**
`ships` is one decision from one A/B over the whole candidate *set*, and the
promotion loop wrote a per-component artifact for each member stamped with that
set-level evidence — so a run that changed one description promoted three, two
of them byte-identical to the incumbent and each carrying `p_value: 0.003906,
candidate_wins: 8`. Unchanged components are no longer promoted, and a changed
one records the set it was judged with. The gate also gained the
`--min-promotion-age-days` re-draw guard its sibling has: at α=0.05 an unguarded
re-promotion loop is a 5%-per-run lottery, and because these artifacts cannot be
re-scored offline, every re-promotion resets the live check's era and discards
the turns accrued against the current one.

**Round 9 gave the §4CW seed veto the outage handling the main arm had.**
`_seed_cmp.transport_excluded` was read nowhere, so a total seed-arm outage
printed a perfect tie, **suppressed the veto and promoted** a candidate that
loses to the hand-written seed at p=0.0000 when healthy — and a partial outage
did the mirror, manufacturing the veto from 5 surviving pairs and refusing an
honest promotion. A veto is a refusal to ship, so an underpowered one is not the
safe direction: it welds the chain to whatever is already live, which is the
ratchet the seed arm exists to prevent. `--allow-seed-loss` overrides it and is
recorded in the artifact.

**Round 10 found that scoping the live comparison de-randomized it.** Round 8
scoped the treatment arm to the live artifact's sha and left control unfiltered,
so treatment became a time window against a control arm of all history —
measured, a contemporaneous KEEP (p=0.6238) read as a REVERT (p=0.0148) that
`--revert` acted on. A control turn now carries the sha of the artifact it was
*withheld*, and both randomized arms are scoped to one era.

**Round 8 found the guard whose own message said it could not be
overridden.** `run_gepa`'s power guard set `candidate_ships = False`, and forty-
five lines below `--allow-insignificant-ship` read that same `False` as "the
discordant pairs were too few" and set it back. Driven: the guard printed
*"Nothing ships — --allow-insignificant-ship does NOT override this"* and the
next line promoted. Its only pin was an AST grep for two strings that the defect
leaves in place — and so was my first replacement, whose docstring claimed to
drive `main()` with the outage and the flag and did not. A one-line mutant
survived the full suite and promoted on 5 pairs. The pin now really drives it,
per arm. Two more, both retire-side: `recheck_gepa_incumbent` reported the exclusion
accounting for the *recorded* gate block and never for the run it had just
performed (a tier that lost 40 of 45 examples to an outage printed a retire
recommendation byte-identical to the healthy run), and `live_check` pooled
treatment turns across artifact SHAs — retiring a healthy artifact on the
evidence of the one it replaced. Both randomized arms are now scoped to the live
artifact's era — a control turn carries the sha of the artifact it was
*withheld*, because scoping only treatment made it a time window against a
control arm of all history and turned a contemporaneous KEEP (p=0.6238) into a
REVERT (p=0.0148). A turn carrying no sha at all is dropped from both arms for
the same reason: it predates the era stamp, and exempting it exempted the
control arm alone. The scoping — and when the artifact exists but no sha can be derived from it
(truncated JSON, a missing or empty `optimized_instruction`), the script refuses
rather than silently falling back to the pooled arm, because that fallback
reached `--revert` and retired a healthy artifact on the evidence of the one it
replaced.

**Round 7 found that round 5's fix could not fire.** `_unreached` matched
`failure_reason` against a list of exception NAMES, and the names were aiohttp's
— this codebase uses httpx exclusively, and no httpx exception subclasses
`ConnectionError` or `OSError`, so only `ReadTimeout` could ever match. Driven
end to end through the real `run_gepa.main()` with identical prompts and a
6-call `ConnectError` outage: **PROMOTED**. A library matching its caller's
exception names is guessing at someone else's dependency; `_run_one` catches the
exception, so it now sets a marker and `_unreached` reads it. An unknown
exception type is excluded correctly, and a `failure_reason` a *runner* produced
— a grading verdict, real evidence — is not.

Round 7 also brought `run_gepa.py` up to the meaning it decides on (it printed
the tier size beside a paired delta, recorded none of the accounting the
re-check branches on, and had no power guard — a 45-call outage left 5 usable
pairs and it shipped), stamped `ab_eval.GATE_METRIC_VERSION` into both gates'
`gate_arm` so the promoted-artifact invalidation rule has something to fire on,
and **reverted round 3's control-path change**: `arm_for() == ""` means *not
enrolled for this request*, never "assigned control". Treating it as control
withheld the artifact from every un-enrolled turn under `traffic < 1` and
inflated `live_check`'s control arm to the point of returning REVERT (p=0.0195)
where the real randomized comparison says KEEP (p=0.2485).

**Round 5 carried the lesson upstream, where it should have gone first.**
`optim/ab_eval.compare_prompts` — the FIRST ship gate, the one `run_gepa.py`
promotes on and `recheck_gepa_incumbent.py` retires on — still scored a timeout
or a transport exception as a failed example. Driven on identical prompts with a
6-call outage confined to one arm: delta +0.120, 6 candidate wins, p=0.0156,
SHIPS=True. That is §4DA's founding defect, measured and closed in the sibling
gate and never brought back to its source. Worse than random in the re-check:
its own docstring records that a timeout scores as a failure and the incumbent is
*by construction* the longer-output arm, so the instrument deciding whether to
**retire** a live artifact was biased toward retirement. Excluded now by a prefix
list — a `failure_reason` is also how a runner reports a legitimate grading
failure, and dropping those would discard the evidence the comparison exists to
weigh.

Round 5 also closed six false or dead operator-facing outputs in the loop around
the gate (the miner writing the live pool for a mine the runner refuses;
`--recordings` being inert against absolute `source.file` paths; `--force-supply`
documented smoke-only and promoting; the rejected-candidate file claiming gate
provenance and a promotion date; the re-check reporting the tier size rather than
the pairs it decided on; and two remedy strings naming a script that rejects the
signature they are printed for), and de-duplicated the artifact schema — `delta`
and `paired_delta` were byte-identical, and `raw_*_pass_rate` duplicated the
top-level pass rates.

**Round 4 found the round-2 guard refusing an honest win forever.** `underpowered`
was armed on the union of "the model was unreachable" and "this fixture has no
recorded payload" — and the second is deterministic and identical in both arms.
A pool with 12 missing recordings and a 6-0 sweep on the 48 replayable rows
(`paired_delta +0.125`, `p=0.0156`, both bars cleared) was refused with *"re-run
when the upstream is stable"*, which is a fixed point costing a full optimizer
run each attempt. The guard is now armed on the outage only, and the pre-flight
probes replayability before it spends anything.

Two more of the same family: `recheck_gepa_incumbent.py` still died with
"unknown signature" on any `tool_description.*` before `--artifact` was even
consulted — so the override warning the round-2 artifact reshape existed to
enable could never print, and the test that "verified" the reshape re-typed the
reader's expressions instead of running it. And every decision-facing number —
the A/B line, the rejection sentence, the artifact's `delta` and pass rates —
stated the RAW margin while the gate decided on the PAIRED one, which produced a
rejection reading "the candidate cleared the margin (delta -0.0500, bar 0.02)".

**Round 3 audited that attribution as production code and found the stamp
could lie.** `_note_served` fires when the loader *loads*; whether the artifact
is *applied* is decided two layers up, past the per-tool validator and the
aggregate-inflation ceiling. With 8 individually-valid artifacts summing past
the ceiling, 40 of 40 requests rendered hand-written baselines only while 21
treatment turns carried a served-stamp — and `gepa_live_check` returned KEEP on
two arms whose prompts were byte-identical. Both refusal points now un-stamp.

Two more, same class: the experiment name truncates to 40 chars, which is
collision-free for the 39 static tools and produces **7 collision groups over
the 70 live names** once composed skills are included — colliding signatures are
never independently randomized, so `--revert` on one could retire the other's
artifact. And `arm_for() == ""` means *control* to the experiment framework
while this loader read it as *serve the artifact*; since the arm ring is capped
at 16, a busy moment leaked control turns into the treatment behaviour, in the
one direction that makes REVERT harder to reach. Both fixed. The promotion write
is now staged + `os.replace`, matching the `run_gepa` line its comment cites.

Finally, the tool-description read site is now **attributed**: it called
`tuned_instruction(sig, "")` with no `req_id`, so no trajectory stamp was ever
written and §4CZ's live judge could only ever report CONFOUNDED for the one
optimizer this gate governs. `--revert` was structurally unreachable. See §4CZ
below for what that path now does.

`scripts/optimize_verifier.py:943` is the third and is **not** done — its metric
is a balanced score over two class arms, so it needs a stratified paired test
rather than a straight McNemar. Do not port this change to it unchanged.

### Judging an artifact by what it did in production (§4CZ, 2026-08-25)

The gate above decides promotion on a few dozen held-out examples, offline. It
cannot answer the question that matters once a prompt is live: **did it help
real turns?** Nothing could, because the provenance was computed and thrown
away — `optim/loader.py` derived an sha8 for every artifact it served and
nothing outside that module ever read it. The artifact retired in §4CW served
every planner turn for weeks on a win nobody could reproduce, and that was
invisible for exactly this reason.

Turns now carry `extra.optim_artifacts = {signature: {sha, arm}}`.

**Attribution alone is not enough, and the analysis refuses to pretend it is.**
An artifact is deployed to every turn at once, so comparing turns before
promotion with turns after it is confounded by everything else that changed —
the corpus, the model's load, what the operator happened to ask. So
`tuned_instruction` also honours an optional randomized arm: register an
experiment named `gepa_<signature>` and the **control arm is served the
hand-written baseline** while treatment gets the artifact. Only that shape
supports a causal claim; `optim/live_check.verdict()` returns `CONFOUNDED` for
anything else rather than a number.

Three arms, deliberately distinct: `treatment`, `control`, and `unenrolled`
(served outside any experiment). **`unenrolled` is not a control group** —
pooling it would reintroduce exactly the confounded comparison the arm exists
to avoid.

Nothing changes until an operator registers that experiment. With no context,
or no registered experiment, the artifact serves everything exactly as before.

**The test is Fisher's exact, not McNemar.** The offline gate runs both prompts
on the *same* examples, so its pairs are matched and a sign test is right. Here
the arms are different requests — unpaired — and McNemar does not apply.
Reaching for the gate's statistic because it was to hand would be a category
error that happens to produce a number. Verified against
`scipy.stats.fisher_exact(..., alternative="less")` over 396 random tables,
zero mismatches.

When it cannot reach a comparison, `gepa_live_check.py` also reads the
registry and names **why** no turn was randomized — unregistered, present in
the file but rejected on load, disabled, `traffic: 0`, a non-live `scope`, arm
names this loader cannot act on, or nothing wrong at all and simply too few
graded turns yet. Those are different problems with different fixes, and the
distinction exists because a single "register the experiment" line was being
shown to operators who had already registered it.

`scripts/gepa_live_check.py` reports the comparison; `--revert` retires a
measurably losing artifact by renaming it `…​.retired-live-<UTC>` — the same
move §4CW made by hand.

> ⚠ **The rename does not stop the running agent.** `optim/loader.py` caches
> the artifact text per process and its `clear_cache()` must not be called on
> a live agent, so retirement takes effect only on the next restart
> (`sudo launchctl kickstart -k system/com.local.ghost-agent`). Until then
> every planner turn keeps using the retired artifact and `activation_stats`
> keeps counting it as applied. The script prints this; do not read
> `RETIRED ON DISK` as "no longer serving".

`--revert` acts **only** on a `REVERT` verdict, so the flag cannot override the
refusal to conclude. Retiring removes a *prefix*: the read site prepends the
artifact to a production prompt, so nothing replaces it.

> ⚠ The name matters: `core.experiments._NAME_RE` is `^[a-z][a-z0-9_]{0,39}$`
> and `_spec_from_dict` **silently skips** a spec whose name fails it. The
> first version of this asked for `gepa.<signature>` — dots — so the
> experiment could never be registered, every turn was `unenrolled`, and the
> analysis could only ever say CONFOUNDED. Use
> `optim.loader.experiment_name()`; do not write the name by hand.

**And a rate cap on promotion.** `run_gepa --min-promotion-age-days` (default 7)
refuses to re-promote a signature whose live artifact is younger than that,
checked with the other pre-flights so a capped run costs nothing. Each run
draws a *fresh* candidate, so repeated runs against a slowly-growing holdout
are repeated draws at the same gate: at the measured accrual — **0.62
private examples/day** averaged over the corpus's 50 calendar days (31 private examples,
2026-07-07 → 2026-08-25), and **0.21/day over the trailing 14 days**, with the newest private
example dated 2026-08-14 — a weekly cadence re-decides on essentially the same evidence, and even the
§4CY gate's 1–3% per-run false-promotion rate compounds to ~0.5–0.8 over 52
draws. Spacing promotions converts that back into a per-run number.

> ⚠ As of writing there are **zero attributed turns** — attribution landed
> after the running agent started, and `planning.decompose` cannot re-run until
> 19 more private examples accrue (31 of the 50 required). At the whole-history
> average (0.62/day) that is ~31 days; at the trailing-14-day rate (0.21/day)
> it is ~90. Read the whole-history figure as a **ceiling on the pace**, not an
> estimate (`traffic-gated-clocks`). `gepa_live_check.py`
> correctly reports `INSUFFICIENT` rather than inventing a verdict. This is
> instrumentation waiting for data, and it should be read that way until the
> first randomized turns accumulate.

### Resolution is not power (2026-08-25)

The refusal above asks whether the metric can *represent* the threshold: with
`n` private examples the smallest non-zero delta is `1/n`, and a `--min-delta`
finer than that is a threshold the gate cannot express. It says **nothing**
about whether an observed difference is distinguishable from noise, and until
2026-08-25 nothing else did either. `ab_eval.compare_prompts` decided the
whole question with:

```python
cmp.candidate_ships = cmp.delta > min_delta
```

A margin, and no significance test. With the guard forcing `n >= 50` at the
0.02 default, the smallest shipping swing is **2 examples out of 50** — and
under the null, a candidate no better than the incumbent, that fires **25-40%
of the time** depending on how many pairs disagree. Run weekly and unattended,
a spurious promotion is a certainty inside a year.

It was also **asymmetric**. The §4CW seed-arm veto already required McNemar
`p <= 0.05`, which needs at least 5 discordant pairs all one way (6 under the
two-sided test it was originally written with); shipping needed two examples. Refusing a promotion took a landslide while making one
took noise — a gate calibrated on the wrong statistic in one direction only.

Promotion now requires the margin *and* support from the discordant pairs at
`ab_eval.SHIP_ALPHA` (0.05, **one-sided** exact McNemar, ties excluded because
only the examples where exactly one arm passed carry information). Both
directions read that constant — the ship side one-sided toward the candidate,
the §4CW veto one-sided toward the seed.

> **One-sided, and it matters.** The first version was two-sided, which spends
> half of `SHIP_ALPHA` on a tail the ship rule can never enter — the direction
> is already fixed by `delta > min_delta`. The realised false-promotion rate
> was 0.011–0.020 rather than the 0.05 the constant advertises, and the cost
> was paid invisibly in power: at n=50 a genuinely +10pp better prompt shipped
> **18%** of the time instead of 28%. A gate that discards four real
> improvements in five is not conservative, it is broken in the direction
> nobody checks.

> **The margin is largely subsumed.** One-sided significance needs 5 discordant
> pairs all one way, so it already implies `delta >= 5/n`. That exceeds a 0.02
> margin for any `n < 250`. On every corpus this project has, **significance is
> the binding constraint and `--ab-min-delta` decides nothing on the ship
> path** — though it still governs the pre-flight resolution refusal and the
> seed veto. One flag, three meanings, one of them currently inert.

There is now one implementation of the statistic *inside the GEPA gate* —
`ab_eval.mcnemar_p`. There were **two** inline copies: `run_gepa.py`'s seed arm
and `recheck_gepa_incumbent.py`'s. (Not three — the library had none, because
`p_value` did not exist before this change.) The repo holds other exact-McNemar
implementations in the ablation runners and `evolve/evaluator.py`; they are
outside this gate and are not covered by that claim.

`p_value`, the discordant-pair counts, and `ship_alpha` are recorded in the
artifact's `gate` block. A promotion whose record cannot answer *"how many
examples actually moved?"* is unauditable, which is how the artifact retired in
§4CW served every planner turn for weeks on a win nobody could reproduce.

**The cost, stated plainly:** a 4-0 sweep is `p = 0.0625` one-sided and does
not ship. The smallest possible evidence misses the bar, so a genuine
improvement on a small signature corpus can be refused. That is what
`--allow-insignificant-ship` is for — it lifts the significance bar only,
never the margin, and stamps `significance_overridden: true` in the artifact.
The refusal message distinguishes the two cases, because *"clear the margin"*
and *"collect more evidence"* are different instructions.

> ⚠ At the time of writing `planning.decompose` has **31 private examples** and
> therefore cannot run at the 0.02 default at all — the resolution guard
> refuses first, needing 50. Nineteen more are required. At the whole-history
> **measured** rate of **0.62/day** (31 private examples over the corpus's 50
> calendar days, 2026-07-07 → 2026-08-25) that is **~31 days**; at the
> trailing-14-day rate of 0.21/day it is ~90. Two earlier versions of this
> note were wrong in the same direction — ~0.8/day and ~24 days by applying
> the *nominal* `--private-pct 30` to the keyed rate (the realised share is
> 25%), then 0.65/day and ~29 days by counting 48 day-directories as 48 days
> when the span is 50. Accrual is also bursty, so treat a month as a
> **floor, not an estimate** (`traffic-gated-clocks` — every pending verdict
> here is gated by ~3.5 turns/day of real traffic). The significance
> requirement is preventative for this signature, not a change to a run that
> is currently happening.

> ⚠ **This fix landed in ONE of three ship gates.** Neither
> `scripts/optimize_verifier.py` nor `scripts/optimize_tool_descriptions.py`
> imports `compare_prompts`; each carried its own
> `ships = ... and delta > args.min_delta`, margin-only, defaulting to 0.02,
> with the resolution guard already in place — i.e. both sat in exactly the
> arithmetic described above. The paragraph further up saying *"all three
> optimizer runners now REFUSE to run when a private tier cannot resolve its
> own threshold"* is true and invites a parity that did not exist for
> significance.
>
> **`optimize_tool_descriptions` is now done — see §4DA below.** Its
> per-fixture score is binary and its two evaluations replay the same fixture
> list in order, so McNemar ports unchanged. `optimize_verifier` is **still
> outstanding**: its metric is a balanced score over two class arms, so it
> needs a stratified paired test, and porting this change to it unchanged
> would give a number for the wrong question.

> The Phase 2b runner was the one that still had no such check, and its
> private tier is the coarsest of the three. Its tier is hashed per *request*
> and one request emits many fixtures, so the realised private share is not
> `--private-pct`: measured 2026-08-04 on the real mine, **13 of 65 positives
> are private (20%, against 30% requested)** — a step of `1/13 = 0.077`
> against a `--min-delta` of 0.02, i.e. one flipped replay decides a run
> costing `iterations × len(public)` main-model calls. `--smoke` is exempt
> (it evaluates the incumbent and ships nothing).

> Measured 2026-08-04 at the real `--private-pct 30` default. An earlier
> version of this page claimed 7 cases / 49 trials / 0.0455 → 21 / 155 /
> 0.0132; none of those six figures reproduced when re-measured. Only 0.0833
> was ever right.

The join that decides which mined turns count as "clean" is keyed on the
**claim**, not the request. `request_id` is per-turn — 348 distinct ids over
12,047 live records, with a single `"SYSTEM"` id holding 9,780 — so of the 62
cases a request-level join excluded, **43 were wrong**: `['REFUTED',
'CONFIRMED']` inside one request is the escalation signature (the cheap judge
false-refutes, the main model overturns), meaning production did *not* refute
that turn. 16 more were dropped on the code auditor's verdict about a
different claim, and 3 sat in the shared `"SYSTEM"` bucket. Per claim, taking
the last verdict: 106 candidates → **86 kept, 20 dropped**, each annotated
with its real production verdict.

The pool is regenerated with `scripts/verify_bench.py --refresh-mined`.
Before that existed nothing in the repo could rebuild it, so the shipped
artifact silently carried whatever extraction and redaction bugs were live on
the day it was minted. The refresh **refuses to write** a zero-yield mint or
one that shrinks the pool by more than half (`--force` overrides): silent
extraction failure is this pipeline's characteristic bug, and overwriting the
durable pool with its output takes the private tier back to 4 cases.

> **Tier caution.** `optimize_verifier.py` trains on the **public** tier of
> this same pool, and `verify_bench.py` loads the pool by default — so a
> post-optimization bench run on the default `--tier all` measures partly on
> cases the optimizer saw. Use `--tier private` for a clean measurement; the
> loader now prints how many public-tier cases it pulled in.

Mined cases are redacted before persisting (recording day-files are unredacted
by design) and live at `$GHOST_HOME/system/eval/verify_bench_cases_mined.jsonl`
— outside the repo, because they derive from real user turns. `--no-mined`
reproduces the old seed-only pool.

**Escalation axis — CLOSED 2026-08-04.** The bench's chat client defined no
critic/worker route, so `Verifier._escalate_refute` returned immediately: the
bench scored the raw cheap judge while production scores judge+escalation.
Re-measured before fixing, two ways: joining recorded verify prompts on the
claim across `$GHOST_HOME/system/llm_recordings` (2026-07-30..08-04) and
reading which model served each verdict gives **42 of 50 (84%)** cheap-judge
refutes overturned by the main model; the durable log's `GhostAgent` lines over
a longer window give **80 / 99 = 81%** — the journal's figure, reproduced. (A
naive `grep` of the log says 89%: the OVERTURNED line is a `WARNING` mirrored
to the `GhostStream` logger while "verdict stands" is `INFO`, so warnings
double-count. Count one logger.)

Now closed on both sides:

* `verify_bench.EscalatingChatClient` gives the bench the production topology —
  `route()` to the judge, `chat_completion()` to the main model, truthy
  `worker_clients`. `scripts/verify_bench.py --main-base-url <main>` selects it;
  verified live end-to-end, where a refuted trial shows the leg sequence
  cheap→cheap→main→main (two-stage on both legs) against 15 s / 24 s runtimes.
* Every report records `provenance.escalation` (arm, kill-switch state, cheap
  route, both endpoint/model identities), and `score_trials` emits
  `fpr_raw_judge` **or** `fpr_escalated` — there is no bare `fpr` key any more,
  so two arms cannot be silently compared. The raw arm's rendered report says
  "NOT a production FPR" in the headline.
* `optimize_verifier.py --escalate gate` makes the SHIP DECISION
  production-equivalent (see above), and the arm is written into the promoted
  artifact.

**Both directions — closed 2026-08-04, same session.** The CONFIRM escalation
landed while the above was being written, and it re-opened the same gap in the
other direction: it fires only on `high_stakes=`, which `run_trials` never
passed, so it was structurally dead in the bench. Bench cases now carry a
tri-state `high_stakes` field (`None` = derive, explicit bool pins), derived by
running production's own `looks_like_tool_error` over each segment of the
packed evidence digest — segmented, because that sniffer scans only the first
120 chars and a blob check therefore sees just the first tool's head (mined
pool: 14/86 segmented vs 10/86 blob). Derivation is resolved per trial *after*
the fault, so `silent_failure` — a tool error under an unchanged success claim
— exercises the direction, which is the only thing that reaches it from the
seed set (0 of 21 seed cases are naturally high-stakes). The arm now has four
values (`raw_judge`, `judge+escalation(refute)`, `judge+escalation(confirm)`,
`judge+escalation`), a new `false_confirm_actionable_{raw,escalated}` metric is
keyed on the confirm direction, and `metrics["escalation_events"]` counts what
actually fired. `GHOST_VERIFY_ESCALATE_CONFIRM=0` is checked by A/B — same
verdict, same confidence, same call sequence as before the feature — rather
than asserted.

See `docs/core/verifier.html` for the full two-direction table and the metric
keying rationale.

**Where the escalation metrics actually live (2026-08-04).** `escalated_overturn`
and `confirm_withheld` were written into `VerifyResult.to_dict()`, which has zero
production callers — 160 "OVERTURNED" log lines, zero occurrences anywhere under
`$GHOST_HOME/system/`. They are now appended to
`$GHOST_HOME/system/verifier/escalations.jsonl` at the point the escalation
resolves (inside `_escalate_refute` / `_escalate_confirm`), which is the only
place that covers BOTH delivery paths: on the streamed path the trajectory is
written in the SSE drain *before* the verdict is even spawned, so no
`turn_facts` stamp could carry it. Upheld outcomes are recorded alongside
overturned ones — the watch metric is a rate. Kill switch
`GHOST_VERIFY_ESCALATION_LOG=0`. The same measurement pass showed the REFUTE
escalation should NOT be extended to `verify_code_output`: all 7 live code-path
cheap refutes were upheld by the main model on two independent replays (14/14),
against 84% overturned on the claim path. Wired but default OFF behind
`GHOST_VERIFY_ESCALATE_CODE_REFUTE=1`.

### Stack audit (2026-08-04) — what was silently not running

Six parallel fresh-eye reviews over the whole self-learning stack found ~100
defects past a green 10.7k-test suite. The generalisable lesson, and the
highest-value fix, was not a logic bug:

**`type(x).__module__.startswith("ghost_agent")` is always False in
production.** The launcher runs `python -m src.ghost_agent.main`, so modules
are `src.ghost_agent.*`; the test suite runs `PYTHONPATH=src`, so they are
`ghost_agent.*`. That guard was used at six sites to mean "not a test double",
and it silently disabled five subsystems on the live agent — including
failure-cluster distillation and the outcome-gated lesson prune, both of which
§3 recorded as `live`. Use `utils.component_guard._is_real_component`, which
accepts both shapes; `tests/test_component_guard.py` fails on any new
occurrence.

Other corrections that changed what the learning loops actually learn from:

* **Episode outcome labels** were `"error" not in ai_text[:80]` — 96.5% of the
  live store read success, and all nine "failures" were false negatives. That
  label feeds the LLM that mints playbook lessons.
* **Self-play was writing the production calibration corpus** and the
  competence prior that the confidence composite reads on real user turns.
* **Infra outages were charged to the agent** as genuine self-play failures,
  with durable consequences (mastery flips, cooldown doubling, lessons minted
  from an outage banner).
* **19% of the GEPA train set** was reflection records presented as gold
  answers (96 of 506 PASSED trajectories, re-measured 2026-08-04). The fix is
  per-FIELD, not per-record: a reflection's `final_response` is a DIAGNOSIS
  block and is blanked; its `planning_output` is a revised PLAN and is kept.
  Filtering whole records instead cost 100% of the plan targets, because
  `planning_output` is populated on reflection trajectories and **nowhere
  else** (157 of 157 live). Consequence worth naming: `run_gepa.py` keeps only
  examples carrying a signature-output target once there are ≥20, so a
  `planning.decompose` run now trains and ship-gates on **96 examples that are
  100% reflection-sourced**, discarding all 410 clean user turns. Defensible
  (a revised plan is a plan) but not what "19% contamination removed" implies.
* The **activation counter** — the instrument built to catch exactly this class
  — counted artifact LOADS, not applications, so rejected artifacts read as
  healthy.
* The **generic GEPA metric was recall-only**, which makes VERBOSITY the
  optimum: a token soup scores 1.000 against the gold it padded and still
  0.250 against a gold it never addressed, while a terse correct subset scores
  0.333. It is now token **F1** (`scripts/run_gepa.py::_overlap`), which
  inverts that ranking (0.367 vs 0.500 on the same 87-token gold).

#### The metric change invalidates the one artifact it promoted

`planning.decompose` was optimized and A/B-gated on 2026-07-29 **under the
recall-only metric** — the only artifact that loop ever promoted. Re-scored
2026-08-04 on the same hash-stable 28-example private tier, both arms at
temperature 0 / no-think against the live upstream:

| metric | seed | promoted artifact | delta | verdict |
|---|---|---|---|---|
| RECALL (the metric that promoted it) | 0.429 | **0.857** | +0.429 | ships |
| F1 (the metric now shipped) | **0.500** | 0.071 | −0.429 | rejected |

The promoted prompt's outputs run a median **111 distinct tokens against a
32-token median gold** — 3.5×, exactly the verbosity the old objective paid
for. The recall column also *reproduces the original promotion* (journal:
0.45 → 0.80), which is what makes the F1 column evidence rather than noise.

~~Read this as **correctness-of-record, not a live emergency**: neither metric
measures plan QUALITY, and the read-site is dark (no `--use-planning` on the
live exec line, activation counter 0). The artifact is kept, not deleted.~~

> ### ⚠ THE PARAGRAPH ABOVE WENT FALSE ON 2026-08-21, AND IT IS WHY NOBODY RE-OPENED THIS
>
> It was true when written (2026-08-16: zero planner calls recorded on 08-15,
> 08-16, 08-17). Then `--use-planning` was added to the launcher exec line —
> `start-ghost-agent.sh.bak-20260807` does not carry the flag,
> `.bak-20260821` does — and the read site fired on **every planner turn**:
> 24 planner calls on 08-21, 77 on 08-22, 58 on 08-23, each carrying the
> 3314-char artifact. **A launcher edit silently armed a read site this
> document certified as dark, and the certification is what suppressed the
> urgency.** `launcher-flag-drift` × `gepa-promoted-artifact-invalidation`.
>
> **Resolved 2026-08-24 (§4CW): the artifact is RETIRED, not kept** — now at
> `system/optim/planning.decompose.json.retired-4cw`, with the measurement
> and a one-line reversal stamped into the file.
>
> ⚠ **But NOT for the reason first recorded, and the correction matters.**
> The initial write-up said "measured worse than the seed" (−0.1220,
> p = 0.0059). Re-measured under a corrected metric the sign **flips**:
> seed 0.4959 vs artifact **0.5041**, delta **+0.0081**, McNemar
> **p = 1.0000**. The artifact is *indistinguishable* from the baseline.
> token-F1 had been grading a **two-field prediction (`plan` + `rationale`)
> against a one-field gold** — `build_trainset` never stamps `rationale` —
> so precision was capped by construction and the more a prompt invested in
> the ungraded field the worse it scored. Its *recall* was better all along
> (0.366 vs 0.294, p = 0.005 in its favour).
>
> It is retired for having **no measured win** while running on every
> planner turn — it buys nothing — and because its terminal directive
> *"Output exactly `### plan` and `### rationale` with no extra text"* is
> prepended ahead of *"Return ONLY valid JSON"* under a JSON grammar: a live
> format conflict. On 8 real recorded planner payloads replayed in the
> production regime, 7/8 parsed with it against 8/8 without.
>
> **And the cause was structural**: `run_gepa`'s gate compares each candidate
> against `_live_incumbent()` — the PREVIOUS artifact — never against the
> hand-written seed. 2026-07-29 artifact 0.071 → 2026-08-07 candidate 0.393
> (+0.321, a real improvement, correctly promoted) → seed, never in either
> comparison, **0.496**. Every promotion was honest and the chain still
> ratcheted away from the thing it should have been beating. The gate now
> runs a third SEED arm and refuses a candidate that loses to it.
>
> Verified live after the 2026-08-24 20:35 restart: planner system prompts
> dropped 7420 → 4104 chars (the difference is the artifact plus its join),
> zero `GEPA: loaded tuned instruction` lines since, and the in-process
> counter reads `planning.decompose: no artifact (baseline)`.

Re-promoting anything here would still require a bench that grades plans
rather than token overlap — that part of the original note stands.

The full list, including ~12 clusters deliberately left unfixed with reasons,
is `PROJECT_JOURNAL.md` §4J.

#### Arming a never-run subsystem is a deploy, not a fix

The module-guard correction made five never-executed subsystems live at once,
and that turned out to be the more dangerous half of the fix.

`prune_low_utility` — destructive, unattended, vector-twin deleting — ran for
the first time in its life and **destroyed 13 lessons across two REM cycles**,
one of them scoring `retrievals=277 succ=77 fail=32` (a 70%-success lesson
dropped as "low utility"). Its archive-before-delete safety net had been
written four minutes *after* the live process started, so that process never
loaded it. **Check module CONTENT HASH against what the process loaded (§4BN R32/R33: mtime was tried and false-fires on a byte-identical restore — the agent now warns by itself via `audit_source_newer_than_process`) before believing a fix
is deployed.**

It is now **off by default** (`GHOST_SKILL_PRUNE=1` to enable), and that is a
calibration verdict rather than caution:

* the cutoff is a *relative* bottom quartile, so it always finds victims
  however good the playbook is — quality never satisfies it;
* failure-distillation mints lessons at utility ≈0.77 against a measured live
  cutoff of 1.1183, so every distilled lesson is structurally guaranteed to be
  deleted once it reaches `min_retrievals`. Two subsystems spending LLM time
  fighting each other, both "working as designed".

The archive now **fails closed** — an unwritable archive aborts the operation
and deletes nothing. A fix whose only purpose is recoverability must not
proceed when recoverability is what failed; the first version warned and
deleted anyway, losing seven more lessons in a probe. The same
archive-before-delete invariant now covers
`retract_lessons_from_trajectory` (four live call sites), which had been
deleting with no record at all. Quarantined rows are exempt from the prune:
quarantine stops their retrievals accruing, so they decay into the bottom
quartile *by construction*.

**Recovery precedent:** 5 of the 13 were reconstructed from
`GHOST_LLM_RECORD` day-files — rendered prompts carry
`TRIGGER/ANTI-PATTERN/CORRECT-PATTERN`, so prompts written before the prune
contain the lesson bodies. Before declaring data unrecoverable, enumerate
every store that ever *rendered* it.

Same batch: journal-mined self-play replays real past user messages **verbatim**
against the real toolset, and the live stash held one instructing
`postgres_admin` to run `SELECT 1; DROP TABLE web_order_line_options_old;` —
un-replayed, at a 75% selection probability. `_is_unsafe_challenge` now refuses
destructive shapes and the "run this exactly, do not modify it" framing at
synthesis time. Any subsystem that replays recorded user input needs a content
gate: the recording is trusted input from a context where a human was present,
and the replay is a context where none is.

### Tool ONTOLOGY analysis (§4F Phase 2b+, 2026-08-05)

Phase 2b optimizes description PROSE. The 0.772 ceiling check
(44/57, 2026-08-03) says that may be the wrong lever: the misses cluster
into specific pairs (browser↔file_system, execute→file_system,
manage_projects over-selected, 3 no-tool stalls), and a pair confused in
BOTH directions is a boundary problem no rewording fixes — it just moves
the error to the other direction.

`ghost_agent/optim/tool_ontology.py` + `scripts/tool_ontology_report.py`
measure the two structural questions:

* **Confusion** — aggregate replay rows (dump them with the runner's new
  `--confusion-out`) into a matrix and classify each hot pair as
  `merge_or_redraw` (bidirectional), `describe` (one-way — Phase 2b's
  real target set), or `missing_affordance` (no tool called at all).
* **Sequences** — consecutive tool-call n-grams over the trajectory
  corpus, ranked by `steps_collapsed × cohesion`. Depth is the agent's
  strongest measured failure predictor (17.8% at step 1 → 60.6% at 12,
  §4H), so collapsing the common path attacks the failure RATE, not just
  latency. Support counts DISTINCT turns (one grind session must not
  mint a proposal); cohesion counts occurrences whose calls share a
  TARGET, with enum fields (`operation=read`) and the redaction sentinel
  excluded and containment matching so
  `path="app.py"` ↔ `command="python3 app.py"` counts as one target.

First live run: `file_system` runs dominate (787 pair-occurrences over
107 turns, cohesion 0.73; ×4 runs would collapse 1431 steps) — a batch
affordance gap. Read-only: it proposes, promotion stays operator-gated.
See `docs/algorithms/tool_ontology.html`.

**Experiment isolation.** Fixtures now exclude turns whose prompt context
was mutated by a live A/B treatment (`core.experiments`), counted as
`experiment_context_excluded` — the optimizer replays payloads verbatim,
so a steered turn would tune descriptions against a context only one arm
sees. `--include-experiment-context` overrides for a post-experiment
re-mine.

### Tool-description optimization (§4F Phase 2b, 2026-08-01)

Tool descriptions are the second always-live GEPA surface. Both halves
of the loop exist; the GEPA run itself waits for fixture supply
(~2-3 days of `GHOST_LLM_RECORD` capture).

* **Miner** — `ghost_agent/optim/tool_fixtures.py` +
  `scripts/mine_tool_fixtures.py` mine tool-CHOICE fixtures from the
  recording day-files per the audited 2026-08-01 contract: **era
  filter** `ts >= 2026-07-31T19:15` LOCAL (the native-prompt split +
  honest-failure rule changed prompt bytes and label semantics before
  that); choice records = reassembled stream records with
  `payload.tools` AND structured `message.tool_calls` (never
  content-parsed); **ground truth** joined from
  `TrajectoryCollector.iter_trajectories()` (corrections overlay
  applied — measured 2026-08-04, 214 of 1488 trajectories read a
  different outcome after the overlay and for 212 of those the on-disk
  line says `unknown`, so the sidecar is the ONLY source of a label;
  reading the raw JSONL would discard them as unlabeled)
  on recording `request_id` == `Trajectory.session_id`; the `SYSTEM`
  request sentinel (idle/background work, >80% of records) is excluded
  — no per-request trajectory exists to join. Polarity: clean PASSED =
  positive; FAILED = negative; **honest-failure turns (PASSED with a
  failed tool call) are EXCLUDED** — the choice signal is ambiguous.
  Tool results pair via ordinal-consecutive records in the same
  recorder session+request, extracted **from the `<tool_response`
  marker onward** — the volatile `<system_state_update>` block is
  prepended to the same user message and must never leak into fixtures
  (the first live mine had 79% contaminated previews before this); a
  same-request background occupant (context-shield summarizer) at
  ordinal+1 yields a lost pair, never a mispair. The mine is ONE
  streaming pass holding only light fixtures (full records are ~100 KB
  each; multi-GB backlogs stay bounded). Public/private tier via
  `holdout_tier("toolfx:<request_id>")` so all fixtures from one turn
  share a tier. Fixtures are LIGHT — a `source` pointer (file,
  session_id, ordinal) lets the eval adapter rehydrate the full
  recorded payload and swap candidate descriptions in (adapter gotcha:
  `RequestState._active_tool_defs_cache` and the XML schema cache key
  on tool NAMES, so build a fresh RequestState per candidate — the
  shipped runner sidesteps this entirely by replaying the recorded
  payload instead of rebuilding a RequestState). The CLI
  prints full drop accounting (no silent caps); exit 1 = supply not
  ready (below `--min-fixtures` OR one-class corpus — zero positives
  and zero negatives both block), exit 2 = no day-files at all.
  `experiment_context_excluded == 0` means "nothing excluded" only when
  `experiment_filter_unavailable == 0` beside it; that flag is resolved
  **eagerly**, before the scan, so it reports an unimportable
  `core.experiments` (or `--include-experiment-context`) even when no
  record survives the earlier filters. `experiment_filter_errors`
  counts turns the filter RAISED on — those are included unchecked.

  ⚠ **`--min-fixtures` means different things in the two tools.** The
  miner's counts ALL fixtures; `optimize_tool_descriptions.py`'s counts
  POSITIVES only (negatives cannot score a tool-choice replay). Both
  default to 200. Measured 2026-08-04: **183 fixtures / 65 positives**,
  so a total-only gate reports "ready" — and atomically overwrites the
  live fixture pool — at roughly 71 positives, while the runner still
  refuses.

  **Closed 2026-08-04 (same day):** the miner now carries
  `--min-positives` (default 200, the gate that actually binds) beside
  the volume floor, so its exit code agrees with its consumer.

  ⚠ **And re-opened by §4DA for one round, one abstraction over.** §4DA
  made the *runner's* supply gate count REAL positives (bench may teach on
  the public side, never grade, and never be the reason a run starts) and
  left the miner's counting bench — so on 2026-08-25 the miner wrote the
  live pool and exited 0 on 403 positives while the runner refused at
  "121 REAL positive fixtures < 200". Both gates are real-only as of
  2026-08-25. The lesson is that a divergence closed between two tools
  re-opens the moment either side's *definition* moves. The flag
  names still differ deliberately — renaming the runner's would break
  every recorded invocation — so the miner's help text names the
  collision at both flags.

  The miner also reports the runner's **resolution** refusal before a
  run is started, because "is it time yet?" should be answerable without
  launching one:

  ```
  Private positives: 13/65 (realised share 20%, requested 30%);
  smallest step 0.077 vs --min-delta 0.02 — TOO COARSE
    → needs ~50 private positives (~250 positives at today's realised
      share) or a larger --min-delta; the runner refuses below this.
  ```

  The realised private share is **measured, not assumed**: the tier is
  hashed per *request* and one request emits 1–40 fixtures, so
  `--private-pct 30` landed 20% on positives. This line is **advisory
  and does not block the write** — the runner owns the refusal, and
  blocking here would freeze the pool at whatever it held on the day the
  tier happened to be coarse, when more supply is precisely the fix.
  Gates fenced in `tests/test_mine_tool_fixtures_gates.py`.
* **Read-site** — `tools/registry._apply_tuned_descriptions` at the
  tail of `get_active_tool_definitions`: a promoted
  `$GHOST_HOME/system/optim/tool_description.<tool>.json` artifact
  replaces that tool's advertised description. Artifact-only (no
  `OptimizableSignature`, mirroring the verifier precedent — the scope
  fence stays untouched); resolution order in-process
  `_TOOL_DESC_OVERRIDES` (offline optimizer) → `optim.loader` artifact
  (activation-counted, surfaced in learning-health) → baseline.
  Validator caps size per tool (`max(6000, 3× baseline)`) AND in
  aggregate (`_TOOL_DESC_AGGREGATE_SLACK`, 20k chars total inflation
  → NO swap at all, all-or-nothing): the per-tool caps sum to ~10× the
  real tools block, so only the aggregate guard actually protects the
  KV-pinned prefix. Rejections warn once per process, not per
  assembly. **KV contract:** the artifact set is scanned once per
  process (and ONLY under an explicit `GHOST_HOME` — the loader's
  `~/ghost_llamacpp` fallback is never scanned, which also keeps
  GHOST_HOME-less test runs from baking a live operator's artifacts
  into the process) and content is loader-cached — warmup and every
  request render identical bytes; deploy = restart, never a live
  cache reset. Copy-on-write: the shared `TOOL_DEFINITIONS` dicts are
  never mutated, and the no-artifact path returns the assembled list
  untouched.

Tests: `tests/test_tool_fixture_miner.py` (era filter incl. local→UTC,
choice detection, polarity + EXIT-CODE-0 gotcha, honest-failure
exclusion, overlay flip, tier determinism, result pairing, round-trip),
`tests/test_tool_desc_readsite.py` (artifact swap, no-mutation,
activation counters, validator, override precedence, identity fast
path, dynamically-appended tools), `tests/test_gepa_optim_reaudit.py`
(incumbent backup + abort-on-failed-backup, gate-judges-the-live-artifact,
resolution refusal in both runners, token-F1 driven end-to-end through
the real gate, over-cap zero-scoring + reflector feedback,
no-cross-candidate-bleed, gate-vs-read-site aggregate agreement,
applied-vs-loaded activation counts, experiment-filter reporting).

### Trajectory-level test-time scaling (§4F Phase 3, 2026-07-30)

Both features are env-gated **OFF by default** per the §3 doctrine (no
unproven layer rides a live turn without a measured win) and read their
switches per call so benches can A/B via env.

**Phase 3a — logit-expectation confidence probe**
(`GHOST_VERIFY_LOGIT_EXPECT=1`): after a two-stage verdict parses
(CONFIRMED/REFUTED only — UNCERTAIN excluded), one bounded score-token
call asks the judge for a single acceptability digit 0-9 with
top-logprobs (`entropy.request_logprobs` handles field selection); the
EXPECTATION over the digit distribution (`_digit_expectation`) is a
continuous p(acceptable) blended into `confidence` (verdict-aligned:
inverted for REFUTED) at weight `GHOST_VERIFY_LOGIT_EXPECT_WEIGHT`
(default 0.25 — the first bench A/B showed w=0.5 drags nearly all
verdicts below the 0.7 actionable gate; clamped [0,1], read per call so
benches can sweep it). Motivation: self-reported
confidence saturates (bench mean-conf ≈ 0.96-1.0 even on wrong
verdicts), so the actionable-confidence gates get no separation.
Verdicts are never changed; probe failure leaves the result untouched;
the raw reading lands in `VerifyResult.probe_score` for observability.
The probe always rides a cheap pool (critic if configured, else
worker) — never the main slot.

> **§4BL update (2026-08-14):** the w-blend described above was RETIRED — the §4BI foreclosure proved no light symmetric blend can move the judge's quantized confidences across the 0.7 gate. The probe's redesigned consumer (the verdict-gated CONFIRM cap) was then NULLed by held-out validation the same day and the probe mechanism RETIRED — the fault signal is within-case (AUC 0.89) and unharvestable by a global threshold. See docs/core/verifier.html and `system/eval/probe_redesign/DECISION_RULE.md`.


**Phase 3b — wobble-band adaptive best-of-N** (`core/tts.py`,
`GHOST_TTS_ADAPTIVE_BON=1`, `GHOST_TTS_BON_K` extra candidates, clamp
1-4): fires at the loop-exit verifier gate ONLY when the verdict is in
the wobble band — UNCERTAIN, or REFUTED below the 0.7 action threshold.
Hard-REFUTED keeps the existing auto-repair path, so the two
regeneration mechanisms never interact. Mechanism (arXiv:2604.16529 /
Agent S3 consensus shape): K alternative standalone finals generated
SEQUENTIALLY on the main model (single-slot box — parallel would
duplicate KV RAM) at diversified temperatures; each reduced to a
deterministic compact excerpt; ONE list-wise comparative judge call on
the cheap pool (never per-candidate independent scores); winner
substitutes `final_ai_content`. The ORIGINAL answer is always candidate
1 and every failure mode (no distinct alternatives, judge error,
unparseable or out-of-range verdict) resolves to it — the pass cannot
make the answer worse by construction. Substitutions log via
`pretty_log("TTS BoN", ...)`.

**Phase 3c — verified-restart: substantially pre-existing.** The
auto-repair loop already restarts generation conditioned on the
verifier's critique, discards the poisoned narration on REFUTED, and is
budget-capped (`_MAX_VERIFIER_REPAIRS`). The §4F delta (conditioning on
a distilled attempt summary instead of the critique) was assessed as
marginal and is deferred.

Tests: `tests/test_verifier_score_probe.py` (expectation math,
env gate, blend alignment, verdict immutability, failure semantics),
`tests/test_tts_adaptive_bon.py` (gates, wobble band incl. enum
verdicts, judge parsing guards, substitution/failure contracts),
`tests/test_stable_prefix_phase3.py` (the §4F flip prerequisite:
enabling `GHOST_TTS_ADAPTIVE_BON`/`GHOST_VERIFY_LOGIT_EXPECT` leaves the
KV-pinned stable prefix byte-identical — same-request payload equality
flags-off vs flags-on, in-request cross-turn pin stability, BoN
candidates built on a copy of the live message list, and a source guard
that the prompt-assembly region reads no Phase-3 env switch).
### What the router writes to disk

| path | what it is |
|---|---|
| `system/router/checkpoint.json` | the trained model. Records `feature_names`, `uses_embeddings` and — critically — `embed_model`, the embedder that produced its training vectors. A checkpoint trained under a different `GHOST_EMBED_MODEL` is **refused at load** and retrained, because width alone cannot tell two 384-d models apart. |
| `system/router/checkpoint.gate_looks.json` | the multiple-looks ledger: which labelled corpus the gate last examined, and which gate configurations it has already asked of it. Prevents re-running the same significance test on the same evidence. Safe to delete; that grants one fresh look. |
| `system/verdicts/<day>.jsonl` | one row per verified answer — `{trajectory_id, verdict, confidence, at, seq}`. **Recording only; nothing reads it.** It exists because every correctness metric already on disk is derived from the same fields the router's own labels are, which makes any quality comparison circular. Join on `trajectory_id`, and take **`max(seq)` per trajectory** — `seq` restarts with the process, so a global max returns a pre-restart row. |

* `router/` uses 18 hand-crafted features **plus** a 384-d embedding of
  the request (§4BQ). The embedder is the vector store's own, already
  resident, so there is nothing extra to download and no egress. A
  checkpoint records which embedder trained it and refuses to load
  against a different one; with no working embedder the router trains and
  serves the lexical-only representation and escalates rather than
  scoring a model it cannot feed.

## Running the eval

The eval has **two tiers**, and it is now trust-aware — a "green" can no
longer be silently meaningless (the old stub-vs-stub compare footgun).

### Tier 1 — the offline invariant gate (no agent, no Docker)

```bash
# Runs in-process in seconds. Regression probes only: cooldown ordering,
# telemetry-off, AND the learning-loop input-integrity + security invariants
# (outcome-labelling exit codes, trajectory schema-drift tolerance, PRM
# junk-outcome skip, browser SSRF guard wired, redaction of special-char
# passwords). Exit 0 iff every invariant holds; use it as the CI gate and
# as a self-audit the agent can run on itself.
python -m scripts.eval_baseline gate
```

### Tier 2 — the online capability baseline (needs a live agent)

This is the number that answers *"is the agent getting better?"* — it MUST be
frozen with `--runner http` against a live agent. A baseline frozen with the
stub is marked untrusted and any compare that involves the stub exits `2`
("NOT A TRUSTWORTHY CAPABILITY COMPARISON"), never a clean `0`.

Use **`--suite capability`** for the live baseline: regression invariants + an
8-task capability set (factual recall, multi-step arithmetic, code tracing,
instruction/format following, structured JSON) + curated probes, all validated
on the agent's *text* reply — so it's fully scorable over plain http with no
Docker.

> **Do NOT freeze `--suite default` over http as your capability number.** The
> default suite also pulls in the challenge-**template** (coding) tasks, and the
> plain http runner can't run their in-sandbox shell-script validator — so they
> score unverified→fail and drag the pass-rate down. First live run of the
> default suite measured 0.679, but that was 19/20 on scorable tasks with 8
> templates counted as un-measured fails.

> ⚠️ **`capability` is PROTECTIVE, not DISCRIMINATING.** It is single-turn,
> zero-tool text Q&A: `mean_tool_calls: 0.0`, and it sits at `pass_rate: 1.000`.
> It proves the agent can still say "Paris" — but it stayed 1.000 green straight
> through five live tool-path bugs (insert_fact hang, flat-0.50 MCTS, native
> tool-call corruption, the `<think>`-strip parse error). For a signal that
> actually exercises tools, use **Tier 3** below.

```bash
# Freeze the capability baseline against a running Ghost on 127.0.0.1:8000.
# Provenance (runner/model/suite) is recorded so a later compare can detect a
# stub or a model/suite mismatch.
python -m scripts.eval_baseline freeze \
    --suite capability \
    --runner http --base-url http://127.0.0.1:8000 \
    --api-key "$GHOST_API_KEY" \
    --model qwen-3.6-35b-a3 \
    --timeout 300 \
    --output "$GHOST_HOME/system/eval/baseline.json"

# Post-learning suite: 5 file-read-shape prompts that score the
# "discover before reading" lesson the Reflector has been producing.
# A live-agent compare against a pre-seeding baseline shows whether
# the lesson is generalising:
python -m scripts.eval_baseline freeze \
    --suite post_learning \
    --runner http --base-url http://127.0.0.1:8000 \
    --api-key "$GHOST_API_KEY" \
    --model qwen-3.6-35b-a3 \
    --timeout 300 \
    --output "$GHOST_HOME/system/eval/post_learning.json"

# Compare a later run to the frozen baseline (after a change / new lessons):
python -m scripts.eval_baseline compare \
    --suite default \
    --runner http --base-url http://127.0.0.1:8000 \
    --model qwen-3.6-35b-a3 \
    --timeout 300 \
    --baseline "$GHOST_HOME/system/eval/baseline.json"
# Exit 0 = no regressions; 1 = a top-level pass_rate drop; 2 = the compare is
# NOT trustworthy (stub involved, or a model/suite mismatch vs the baseline).
```

### Tier 3 — the behavioral (execution-grounded) suite — needs a live agent

This is the **discriminating** signal `capability` isn't. Each task DRIVES the
live agent and then VERIFIES the real side-effect — a file written in the
sandbox, a fact that recalls on a follow-up turn, a row from the actual DB — so
it only passes when the tool did its job. It forces `--suite behavioral` onto
its own runner (`agent_behavioral_runner`) + an `EvalContext` that owns the
agent endpoint, the sandbox path, DB access, and trajectory-metric extraction; a
stub/echo can't verify a sandbox file, so a `BehavioralTask` run under any other
runner scores FAIL ("unverified"), never a soft green.

Why it matters: the first live run reported `mean_tool_calls: 2.40` /
`mean_tool_errors: 0.40` (vs `capability`'s `0.0` / `0.0`) — it actually
exercises the tool paths, and even surfaces recovered tool-strikes on passing
tasks. The five shipped tasks (`eval/behavioral.py`) are grounded regression
catchers — each would have FAILED on the pre-fix agent (e.g. `beh:memory_roundtrip`
times out on the old insert_fact hang; the native corruption strikes). Next step
for full *upward* headroom: add a few genuinely-hard graded tasks so improvement
shows, not just regression.

```bash
# Freeze the behavioral baseline (kept SEPARATE from the capability baseline).
GHOST_HOME=/path/to/ghost_home python -m scripts.eval_baseline freeze \
    --suite behavioral \
    --base-url http://127.0.0.1:8000 \
    --api-key "$GHOST_API_KEY" \
    --model qwen-3.6-35b-a3 \
    --timeout 200 \
    --output "$GHOST_HOME/system/eval/baseline_behavioral.json"
# GHOST_HOME must be exported — the runner reads $GHOST_HOME/sandbox to verify
# files and $GHOST_HOME/system/trajectories for real tool_calls/tool_errors.
```

**Using it as the self-improvement gate.** This is the intended workflow for
recommendation-1 ("make success measurable"): freeze a capability baseline
once, then before any self-improvement change is allowed to affect production
(a promoted lesson, a tuned prompt, a PRM update) run `compare` on held-out
tasks and require a non-regression (exit 0). A mechanism that can't beat — or
at least hold — the baseline hasn't earned the right to change behaviour.

Flag notes:

* **`--runner stub`** — the default; echoes prompts and makes
  non-regression tasks fail. Exists so CI can exercise the pipeline
  without a live upstream. A stub-frozen baseline is marked untrusted and a
  stub compare exits `2`, so it can never masquerade as a real gate.
* **`--runner http`** — POSTs to a running agent over loopback.
  The network guard permits only `127.0.0.1` / `localhost`, so this
  is the only shape of real-agent eval that stays privacy-safe.
* **`--timeout N`** — applied to BOTH the httpx client AND
  `EvalSuite.per_task_timeout_s`. Default 300 s. Template tasks that
  multi-turn against a local Qwen-class model commonly run 80–250 s;
  the earlier 60 s default produced spurious timeouts that made the
  baseline `pass_rate` look worse than the agent's actual behaviour.
* **`--suite post_learning`** — small targeted bank used to
  demonstrate reflection lessons are being picked up by the memory
  bus on fresh turns. Passing means the agent's response contains a
  discovery signal (`list / find / search / locate / verify /
  workspace` keywords); failing means it blindly fabricated a result
  without verifying the file exists.

## A/B: prove the reflection loop actually lifts capability

The loop is wired and its closure is verified in-process
(`tests/test_reflection_loop_closure.py`: a FAILED trajectory becomes a lesson
in SkillMemory), and the dedup set now persists across restarts so it keeps
progressing through the failure backlog. What that test canNOT show is whether
the produced lessons make the agent *better* — that is mediated by the model
and needs a live A/B. Concrete protocol:

```bash
# 0. Start the agent with an EMPTY playbook (fresh $GHOST_HOME or cleared
#    skills_playbook.json). This is the "pre-learning" agent.

# 1. Freeze the pre-learning baseline on the behaviour-sensitive suite:
python -m scripts.eval_baseline freeze --suite post_learning --runner http \
    --base-url http://127.0.0.1:8000 --api-key "$GHOST_API_KEY" \
    --model qwen-3.6-35b-a3 --timeout 300 \
    --output "$GHOST_HOME/system/eval/pre_learning.json"

# 2. Let the agent ACCUMULATE lessons: run real sessions that fail the
#    discover-first behaviour, or leave it idle so the biological watchdog's
#    reflection phase reflects the logged failures into the playbook. Confirm
#    lessons landed: `jq '.|length' $GHOST_HOME/.../skills_playbook.json`.

# 3. Compare the SAME agent (now with lessons) to the pre-learning baseline:
python -m scripts.eval_baseline compare --suite post_learning --runner http \
    --base-url http://127.0.0.1:8000 --api-key "$GHOST_API_KEY" \
    --model qwen-3.6-35b-a3 --timeout 300 \
    --baseline "$GHOST_HOME/system/eval/pre_learning.json"
```

A positive `pass_rate_delta` (with no trust warnings — same model/suite, http
on both sides) is the evidence that reflection lessons are changing behaviour
in the intended direction. A flat or negative delta means the loop is running
but NOT helping — which is exactly the thing worth knowing before investing
more in it, and the reason the whole spine now hangs off this measurable gate.

## Wiring the reflection phase

Wired automatically in `main.py` during `lifespan`. Minimum config
for a custom entry point:

```python
from ghost_agent.distill import TrajectoryCollector
from ghost_agent.reflection import Reflector

async def critique(prompt: str) -> str:
    # max_tokens=4096 is deliberate — Qwen 3.6 is a reasoning model
    # whose hidden `reasoning_content` often exceeds 2000 tokens.
    # A short cap leaves the `content` field empty and the reflector
    # logs "unparseable reflection response".
    res = await llm_client.chat_completion({
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 4096,
        "stream": False,
    })
    return res["choices"][0]["message"]["content"]

ctx.trajectory_collector = TrajectoryCollector()  # writes to $GHOST_HOME/system/trajectories
ctx.reflector = Reflector(
    critique_fn=critique,
    # 120s ceiling: Qwen 3.6 is a reasoning model whose
    # `reasoning_content` phase regularly burns 30-60s before any
    # visible content, AND the post-turn reflect_one path competes
    # with the user-facing turn for the same upstream. 45s was too
    # tight in practice — observed silent timeout on the post-turn
    # path even though the structural promotion fired correctly.
    per_call_timeout_s=120.0,
    model=args.model,
)

# The important part — the composite sink is what CLOSES the loop.
# Without it, reflections just land in JSONL and nothing reads them.
def reflection_sink(traj):
    ctx.trajectory_collector.append(traj)
    ctx.skill_memory.learn_lesson(
        task=traj.user_request[:400],
        mistake=traj.extra.get("source_failure_reason", "failure")[:400],
        solution=(traj.planning_output or traj.final_response)[:1200],
        memory_system=ctx.memory_system,
    )
ctx.reflection_sink = reflection_sink
```

With those set:

* **The biological watchdog** fires reflection every ~40 min on
  recent FAILED trajectories that the user never returned to (or
  whose correction the heuristic gate missed).
* **The real-time post-turn path** (`_maybe_promote_prior_turn_via_user_correction`)
  is automatically active once `ctx.reflector` and
  `ctx.trajectory_collector` are wired — `handle_chat` invokes it
  on every user turn with no extra opt-in. A correction-shaped
  follow-up promotes the prior trajectory and schedules
  `reflect_one` immediately; the lesson typically lands within
  ~10 s of the correction message returning, on a warm upstream.

Each reflection is persisted to JSONL AND to `SkillMemory`; on any
future user turn whose semantic-neighbourhood retrieval surfaces the
lesson (planner pre-fetch at `agent.py:2260`, execution-stage
fetch at `agent.py:2402`), the agent enters the turn already primed
with the corrected plan.

### Plan verification (proposal #6 — grounding the reflection loop)

Reflection used to be the **one** learning path with no correctness
grounding: the revised plan was written straight to `SkillMemory`,
executed-or-not, correct-or-not (unlike self-play lessons, which pass
through `dream._verify_lesson_helpful`). The `Reflector` now takes an
optional injected `verify_fn(traj, plan) -> (verified, note)` (sync or
async, mirroring `critique_fn`). When wired:

* the revised plan is checked against the diagnosed failure before the
  lesson is trusted;
* the reflection trajectory's outcome is upgraded to `PASSED` **only**
  on a verified verdict — otherwise it stays `UNKNOWN` (so the un-wired
  path is byte-for-byte unchanged);
* the verdict + note land in `extra["plan_verified"]` /
  `extra["plan_verify_note"]` and are appended to `final_response`.

`main.py` wires `verify_fn` to an **independent LLM judge** (temp 0,
strict `VERDICT: CONFIRMED|REFUTED` rubric) that asks "would this revised
plan avoid that specific failure?". It runs only on the fire-and-forget /
idle reflection paths, so it adds no user-facing latency. The injection
point also lets a sandbox re-run back the verifier for self-play-derived
reflections (where a validator exists) without touching the driver.
Covered by `tests/test_reflection_plan_verify.py`.

## Cooldown anchor discipline

All idle-triggered phases mirror the same pattern — fail to follow
it and the phase re-fires every 60 s on exception until the idle
window naturally expires:

```python
# 1. Set anchor BEFORE await — a crash mid-run still advances it.
self._last_reflection_at = datetime.datetime.now()
try:
    await reflector.run(...)
finally:
    # 2. Re-affirm in finally — belt AND braces.
    self._last_reflection_at = datetime.datetime.now()
```

`_last_dream_at`, `_last_reflection_at`, `_last_skills_auto_at`,
`_last_prm_train_at`, and `_last_selfplay_at` all follow this shape.
The `test_reflection_biological_tick` and `test_prm_biological_phase`
integration tests exercise this explicitly — the anchor must advance
even when the inner call raises.

## Verified end-to-end (2026-04-24)

Direct functional test against the live agent (upstream Qwen 3.6 35B-A3):

1. **Seeded trajectories:** 2 FAILED (`FileNotFoundError: access.log`,
   `awk: can't open file emails.txt`).
2. **Reflection produced specific diagnoses:**
   * "The file `access.log` is missing from the workspace, so the directory must be listed to identify the correct name"
   * "The `emails.txt` file was not present in the sandbox workspace when the awk command was executed"
3. **Plans were actionable** (3-step sequences starting with `file_system(action=list)`).
4. **`SkillMemory` playbook grew from 1 → 3 lessons.** The skill_mem's
   "🎓 skill acquired — Lesson learned: ..." log fired for each.
5. **Retrieval works on unseen similar prompts.** User sent *"I need
   to parse a logfile and count errors. Just tell me your FIRST step
   in 1 sentence."* — the memory bus hydrated the lesson, and the
   agent replied with **"I'll search for the log file in your workspace
   so I can locate and analyze it"**. That is the corrective behaviour
   the Reflector learned, applied without any weight update.
6. **Post-learning eval:** 3/5 targeted prompts scored as
   discover-first, exposing a measurable generalisation delta.
7. **Default eval suite:** 15/15 (`pass_rate=1.000`) with
   `--timeout 300`. All 8 template clusters (`algo, bash, concurrency,
   data_analysis, python_general, regex_parse, sql, web_automation`)
   completed in 84–246 s — the earlier 0.400 pass-rate was a pure
   timeout artifact, not an agent regression.

## Sandbox image (prerequisite for template tasks & self-play)

Template-cluster tasks (`data_analysis`, `regex_parse`, `algo`, `bash`,
`sql`, `concurrency`, `python_general`, `web_automation`) and the
self-play harness all run LLM-emitted code inside a Docker container.
The container image is `ghost-agent-base:latest`, built from
`sandbox/Dockerfile`. Build it **once per Ghost version**:

```bash
scripts/build_sandbox_image.sh
# → builds ghost-agent-base:latest (~2 GB first run; ~5 min on a warm
#   docker cache) and runs a Chromium smoke test.
```

The Dockerfile bakes apt system packages, the deep-learning pip stack,
and `playwright install --with-deps chromium` at image build time —
self-play can launch browser tasks immediately without burning agent
turns on runtime re-installs.

If the image is missing, the runtime wrapper falls back to installing
everything inside a fresh container on first boot, committing to
`ghost-agent-base:latest` when done. Both paths converge on the
`/root/.supercharged.v2` marker; older images without it are treated
as un-provisioned.

Diagnostic: if the self-play log shows `playwright install chromium`
firing as an agent tool call, the container's Chromium install is
broken. The runtime gate now detects this (marker present + binary
absent) and forces a re-install on next `ensure_running`; if the
behaviour persists, rebuild the image: `scripts/build_sandbox_image.sh`.

## Closing the loop on interactive-session failures (2026-04-26)

The Reflector iterates only `outcome=FAILED` trajectories. Chat turns
ship with `outcome=UNKNOWN` because there's no validator on free-form
chat — only self-play and self-consistency batches produce explicit
FAILED. That made the self-improvement loop *blind* to interactive
sessions: a 70-minute thrash on a misdiagnosed UI bug never produced
a lesson because the per-turn trajectories all stayed UNKNOWN.

`distill/outcome_heuristics.py::classify_chat_outcome` looks at a
just-recorded chat trajectory and promotes UNKNOWN → FAILED when one
of four signals fires:

1. **Runtime abort markers** — `[ATTEMPT_ABORTED_*]` substrings in
   `final_response` (cross-turn loop, thinking-budget cap, n-gram
   loop, …). These markers fire only when an in-band guard has
   already determined the turn was non-productive, so they're a
   strong signal.
2. **Browser selector thrash** — the same selector appears in ≥ 4
   browser tool-call invocations within one turn (atomic ops + every
   `interact` sub-action are counted). This is the exact shape of the
   2026-04-26 webOS incident: identical click selectors fired across
   8 nested `interact` calls.
3. **Repeated identical tool errors** — the same `(tool, normalized
   error message)` pair appears ≥ 3 times in one turn. Errors are
   normalised (whitespace squash, lowercase, leading "Error:"
   prefix stripped) so two textually-similar errors hash to one key.
4. **Browser sequence aborted** — the result text contains
   `⚠ SEQUENCE ABORTED` (set by `op_interact` when a goto fails and
   cascades through the rest of the action list).

The classifier is **conservative**: existing PASSED / FAILED outcomes
are never overruled, three repeats of the same selector is below
threshold, and a single tool error doesn't fire. False positives
flood the lesson store with bad reflections, so the bar stays high.

`apply_chat_outcome_heuristics(traj)` is the in-place wrapper called
from `core/agent.py::_record_turn_trajectory` just before
`collector.append`. It runs after the trajectory is fully assembled;
classification failure is logged at debug and never blocks the turn.
Cross-turn signals (e.g. "the same misdiagnosis appears across 5
turns") need session-scoped state and are deliberately out of scope —
they belong in a future `session_telemetry.py` keyed by `session_id`.

Coverage: `tests/test_trajectory_failure_heuristic.py` (signal
matrix, threshold knobs, no-op on healthy turns, end-to-end
integration with the Reflector).

## Real-time loop closure: user-correction promotion (2026-04-28)

`outcome_heuristics` (above) catches *mechanically-stuck* failures —
selector thrash, repeated tool errors, abort markers. It misses the
dominant interactive-chat failure mode: the agent confidently
produces an answer that's *wrong*, the user pushes back, and we
want to learn from that exchange before the user's next message.

The user's next message **is** the cheapest, most reliable supervisor
for free-form chat. If they're correcting us, the prior turn was
FAILED — by the user's own verdict, no validator required. Two
mechanisms make that signal usable:

### 1. The classifier (`distill/user_correction.py`)

Pure-Python, two-signal predicate. **Promotion requires BOTH** to
fire:

* **Signal A — anchored correction phrase.** Regex anchored at the
  start of `current_user_text`: `no`, `nope`, `wrong`, `actually`,
  `that's not right`, `I meant`, `you misunderstood`, `try again`,
  `redo`, `didn't work`, … A "no" deep inside a sentence does *not*
  count (anchored start guards against discourse-marker false
  positives).
* **Signal B — semantic rephrase.** Token-overlap Jaccard between
  the prior user message and the current one, computed over
  content tokens (stopwords stripped — articles, pronouns, common
  quantifiers, common modal verbs). Threshold ≥ `0.40`. The
  intuition: if the user is re-asking the same question, that's
  strong evidence the prior assistant answer was inadequate.

A single signal alone has too many false positives. *"No, I think
you're right"* is phrase-without-rephrase. *"… and also, what about
X?"* is rephrase-without-phrase. Both signals together catch the
genuine corrections while leaving prosaic follow-ups alone. The
classifier is purely lexical — no LLM call, no embeddings.

* **Affirmation veto (2026-07-05).** Both signals CAN fire on
  praise: *"actually the sort you wrote works great"* opens with a
  correction phrase ("actually") and echoes enough of the request's
  content words to clear the Jaccard threshold — pre-fix that
  promoted a GOOD turn to FAILED and retracted its lesson
  (self-poisoning, the same class the 2026-07-05 chess post-mortem
  cleaned out of vector memory). Now, when both signals fired, a
  clear affirmation (`_AFFIRMATION_RE`: "works great/now/fine",
  "you're right", "looks good", "spot on", …) with **no negative
  marker** (`_NEGATIVE_MARKER_RE`: anchored "no", "wrong",
  "doesn't/didn't work", "broke", "still failing", … — phrase-based,
  no bare "error"/"exception" substrings) vetoes the verdict
  (`is_correction=False`, `confidence=0.0`, `"affirmation-veto"`
  appended to `signals` for audit). Ambiguity resolves toward
  correction: *"No, you're right, …"* still promotes (the anchored
  "no" blocks the veto) — a missed veto only costs a lesson, a
  wrongly vetoed real correction costs the FAILED label.

Knobs (module-level constants for runtime tuning):

| Constant | Default | Meaning |
|---|---|---|
| `JACCARD_REPHRASE_THRESHOLD` | `0.40` | Minimum content-token overlap for Signal B |
| `MIN_CURRENT_TOKENS_FOR_REPHRASE` | `2` | Floor on current-message content tokens (a bare "no" can't fire B) |

Coverage: `tests/test_user_correction.py` (29 cases — phrase
coverage, single-signal guards, anchored start, defensive
normalisation of None / non-string inputs, threshold pinning,
verdict-shape contract, affirmation-veto true/false positives).

### 2. The wiring (`core/agent.py`)

Two new helpers on `GhostAgent`:

* `_stash_trajectory_for_correction_lookup(traj)` — called inside
  `_record_turn_trajectory` right after `collector.append`. Builds
  a stable md5 fingerprint of the response prefix (whitespace-
  collapsed first 500 chars, lowered) and stores
  `{fingerprint: traj}` on `ctx._recent_trajectories_for_correction`.
  Bounded LRU at 32 entries — enough for several concurrent
  conversations without unbounded growth.

* `_maybe_promote_prior_turn_via_user_correction(messages, current_user_text)` —
  called from `handle_chat` immediately after `last_user_content`
  is set. Walks `messages[:-1]` to find the prior assistant +
  prior user, fingerprints the assistant content, looks up the
  cached trajectory. If the classifier returns
  `is_correction=True`, it:
  1. mutates the cached trajectory's `outcome` and `failure_reason`
     **in memory** (so the immediate `reflect_one` call sees them);
  2. appends a record to the corrections sidecar (durable);
  3. drops the cache entry (one promotion per stashed trajectory);
  4. schedules `Reflector.reflect_one(traj, sink, already_reflected)`
     via `loop.create_task` — fire-and-forget, the user turn
     doesn't block on the LLM critique.

The `_pending_reflection_tasks` set on the context tracks in-flight
tasks (each adds itself, removes on done). A done-callback logs
the result via `pretty_log("Post-Turn Reflection", …)` — `ok` with
the diagnosis preview, `no lesson` with the error reason
(`timeout after Ns`, `unparseable reflection response`), or
`failed` with the exception type. **Without that callback the
async task's result is invisible**: a critique timeout silently
produced no lesson and operators couldn't tell the difference
between "loop misfired" and "LLM was slow".

The shared `_reflected_trajectory_ids` set is honoured by both
`Reflector.reflect_one` (real-time) and `Reflector.run` (biological
backstop) so a trajectory reflected via the real-time path is
skipped by the watchdog and vice versa.

### 3. The corrections sidecar (`distill/collector.py`)

Outcome promotions discovered AFTER the original JSONL write land
in **`corrections.jsonl`** at the trajectory tree root, NOT by
rewriting the original line. `update_outcome(trajectory_id, outcome,
reason, source=…)` appends a JSON line; `iter_trajectories` overlays
the latest correction per id on read. Properties:

* The original JSONL line stays byte-identical — the audit trail is
  preserved.
* Last-write-wins on repeat updates for the same id.
* Malformed sidecar lines are skipped without poisoning the overlay.
* Orphan corrections (id not in any JSONL) are silently ignored on
  read.
* `update_outcome` is a no-op when the collector is `enabled=False`
  (mirrors `append`).
* The sidecar is a single file (NOT day-partitioned) — the workload
  is tiny (one record per failed turn) and a single growing file
  lets readers apply corrections in O(corrections) instead of
  scanning every day's directory.

Coverage: `tests/test_trajectory_corrections_sidecar.py` (12 cases),
`tests/test_post_turn_reflection_wiring.py` (12 cases), and the
end-to-end ratchet `tests/test_self_improvement_loop_e2e.py` (2
cases).

### 4. `Reflector.reflect_one(traj, sink, already_reflected)`

Single-trajectory entrypoint that bypasses the iterator path used
by `run()`. Honours the same `already_reflected` dedup set, and
adds the trajectory id to the set **before** the await — so a
concurrent biological-tick `run()` can't double-reflect on the same
trajectory while `reflect_one`'s critique is pending. Sink contract
matches `run()`: invoked once per ok reflection; sink exceptions
are logged at WARNING and swallowed.

Coverage: `tests/test_reflect_one.py` (7 cases).

### Live verification (2026-04-28)

Multi-turn against the running agent on `:8000`:

```
turn 1 user: "Reply with three words exactly: kangaroo trampoline lighthouse..."
turn 1 ai  : "kangaroo trampoline lighthouse"
   → trajectory id=46df2dfe... outcome=unknown
   → fingerprint stashed on ctx cache

turn 2 user: "no, reply with three words exactly: kangaroo trampoline metronome..."
   → classifier verdict: is_correction=True, signals=[phrase, rephrase(jaccard=0.82)]
   → corrections.jsonl record:
     {trajectory_id: "46df2dfe…", outcome: "failed",
      reason: "user-correction signal: phrase + rephrase(jaccard=0.82)",
      source: "user_correction"}
   → TrajectoryCollector overlay yields outcome=FAILED ✓
   → asyncio.create_task(reflector.reflect_one(...))
   → done-callback logs:
     post-turn reflection: ok (traj=46df2dfe): diagnosis='The previous response
     likely included extra text...'
   → SkillMemory.learn_lesson → playbook 22 → 23 lessons
   → similar query "respond with three words exactly" surfaces the new lesson
```

The first attempt of this exact test failed cleanly with
`no lesson (traj=…): timeout after 45.0s` (the model was busy
processing turn 2 on the same upstream). Bumped
`Reflector.per_call_timeout_s` from 45s → 120s in `main.py` —
Qwen 3.6 35B-A3 is a reasoning model whose hidden
`reasoning_content` regularly burns 30-60s before emitting visible
content, and 45s left no headroom when the user-facing turn was
saturating the upstream. After the bump, `reflect_one` completes in
~9s on average and the lesson lands within seconds of the user's
correction message returning.

## Wrong-question detection: verifier alignment + lesson retraction (2026-04-28)

A user trace exposed a triple failure: user asked *"how can I see how many lines of code is a project? just give me the code"*; the agent ran `wc -l` in its own sandbox and replied *"The project has **1,623 lines of code**"*; the verifier returned **CONFIRMED (100%)**; the Perfection-Protocol then saved an **"Optimization Analysis"** lesson into `SkillMemory` based on that wrong answer. Three layers failed in sequence — the model misread the user, the verifier rubber-stamped it, and the opt-prot baked the misread into long-term memory.

The fix touches all three layers.

### 1. Verifier audits user-request alignment

Before: `verify_code_output(code, output, intent)` asked the LLM "does the OUTPUT contain the information the user asked for?" — a check that's true whenever the printed claim is internally consistent with the tool output, regardless of whether the agent answered a different question than the user asked.

After: `verify_code_output(code, output, intent, *, response="")` takes the agent's user-facing reply as a fourth slot. The prompt rubric leads with **constraint satisfaction**:

> Does the user's wording include explicit constraints on the form of the answer? Examples: "just give me the code", "in one sentence", "without using X", "list only the names", "as JSON". If yes, does the AGENT'S RESPONSE satisfy those constraints? **If the user asked for code and the agent returned a number / prose / a result, that is a REFUTED — the agent answered a different question than the one asked, even if the tool output is internally consistent.**

The prompt enumerates the failure shapes explicitly (user asks for code → agent returns a result; user asks for format X → agent ignores it; tool output is a sandbox-internal artefact the user can't actually use) so the verifier LLM has concrete patterns to match. A CONFIRMED verdict requires BOTH the tool output to be sound AND the response to match what the user asked for. The verifier callsite in `core/agent.py` passes `response=final_ai_content` so the rubric has the agent's reply to audit.

Coverage: `tests/test_verifier_user_intent_alignment.py` (10 cases — prompt content invariants, the response slot rendering, back-compat sentinel for callers that don't pass `response`, response-slot truncation, and a rubric-following stub that pins the exact 12:04 failure shape gets REFUTED under the new prompt format).

### 2. Lesson provenance via `source_trajectory_id`

Every `learn_lesson` write now records the trajectory id of the turn that produced the lesson. Persisted on:

* the JSON playbook entry (`source_trajectory_id` field, populated via `build_lesson` and surfaced through `_normalize_lesson` so legacy lessons read back as `""`);
* the vector-store metadata (so `collection.delete(where={"source_trajectory_id": ...})` is the one-liner that scrubs the embedding tier).

Two production writers thread the id:

* **Perfection-Protocol** (`core/agent.py::handle_chat` → `learn_lesson(...)`) uses the current turn's pre-allocated `current_trajectory_id`. The id is allocated at the **start of `handle_chat`** with `uuid.uuid4().hex` because the opt-prot fires BEFORE `_record_turn_trajectory` writes the trajectory to disk; both callsites must use the same id or retraction can't link them.
* **Composite reflection sink** (`main.py`) uses `reflected_trajectory.extra["reflected_from"]` — the *original failed trajectory's* id, not the reflection trajectory's own id. Rationale: the reflection's lesson IS the corrective behaviour for that source failure, so provenance unifies under one id per source-of-failure.

Legacy lessons (written before the schema change) read back as `source_trajectory_id=""`. The empty-string-id case is a **deliberately protected sentinel**: `retract_lessons_from_trajectory("")` returns 0 without touching disk, so a buggy caller passing an empty string can't accidentally scrub every legacy lesson at once.

Coverage: `tests/test_skill_provenance_and_retraction.py` (16 cases across schema, persistence, retraction matching/idempotency, legacy protection, vector-delete `where`-filter shape, error swallowing, and the full poison→correction→retraction integration).

### 3. Retraction on FAILED promotion

`SkillMemory.retract_lessons_from_trajectory(trajectory_id, memory_system=None) -> int` is the scrub primitive. JSON pass under the lock, atomic write of the surviving entries; vector pass via `collection.delete(where={"source_trajectory_id": ...})` (best-effort — JSON is the canonical store). Idempotent. Returns the count removed from the playbook. Logged via `pretty_log("Skill Retracted", …)` so a tail of the agent log makes scrubs visible.

Two callsites in `core/agent.py`:

* **Verifier-driven retraction (preventive)** — when the verifier returns REFUTED with confidence ≥ 0.7, the gate appends the verifier note to `final_ai_content` AND immediately calls `retract_lessons_from_trajectory(current_trajectory_id, memory_system=ctx.memory_system)`. This catches the dominant case at source: the Perfection-Protocol's lesson is on disk, the verifier just disagreed with the response, scrub before the user even sees the reply.
* **User-correction-driven retraction (recovery)** — `_maybe_promote_prior_turn_via_user_correction` calls retract on the prior turn's id immediately after writing the sidecar correction record and BEFORE scheduling reflect_one. The reflection then writes the corrective lesson with the same `source_trajectory_id` (because `reflected_from` is the prior turn's id), so the playbook ends up with the right entry rather than both. Without retraction, the previous-turn's poisoned lesson and the reflection's corrective lesson would coexist in the playbook with no demotion mechanism, and BM25 / vector ranking would still surface the wrong one for some queries.

Both retraction paths are wrapped in `try/except logger.debug` so a retraction failure can never break the user turn. The verifier-driven path runs synchronously inside `handle_chat`; the user-correction path runs synchronously inside the next-turn classifier helper.

### Live verification against the running agent (2026-04-28)

Re-issued the original failure prompt. The agent's response now leads with the command — *"Here's the command: \`find . -type f \\(-name "\*.js" -o -name "\*.html" -o -name "\*.css" -o -name "\*.py" \\) -exec cat {} + | wc -l\`. For this sandbox, the result is **1,601 lines of code**."* — and the verifier returned CONFIRMED (correctly: the user got the command they asked for). The Perfection-Protocol's eager-write gate didn't fire because the response is now > 50 chars (the gate guards against empty replies, not against verbose ones), so no poisoned lesson was written for this turn. The polluted entry from the original 12:04 trace remains in the playbook with `source_trajectory_id=""` (legacy, pre-schema-change) — the protection sentinel keeps it safe from accidental bulk retraction; future opt-prot writes carry provenance and can be scrubbed cleanly.

The non-reproducibility of the failure is itself a partial validation: the agent improved its answer between runs because the polluted lesson surfaced its previously-cached find/wc one-liner in the system-prompt context, and the agent's own planner used it. We can't tell from this alone whether the verifier alignment fix would have caught the original wrong response, so the prompt-rubric audit lives in the unit test (`test_wrong_question_shape_can_be_refuted` exercises a stub LLM that follows the rubric literally on the exact failure-trace inputs and asserts REFUTED).

## Browser `interact` abort semantics

The `browser.interact` op runs a list of sub-actions inside a single
Chromium context. Under the default `stop_on_error=False`, a failed
per-action step (e.g. a click on a missing selector) is recorded and
the loop continues — useful for "try all these selectors, tell me
which ones matched" exploratory flows.

**Navigation failures are the one exception: they always abort the
sequence, regardless of `stop_on_error`.** A `page.goto(...)` that
raises (ERR_FILE_NOT_FOUND, ERR_CONNECTION_REFUSED, DNS failure, …)
leaves Chromium on an error page; every subsequent click/fill/
extract_text would just wait the full per-action timeout for elements
that don't exist. Before the fix a 54-action sequence whose first
goto 404'd hung for ~108 minutes (54 × 120 s) before the outer
subprocess timeout fired.

The fix: `op_interact` in the runner catches the `goto` exception,
records `aborted_sequence: True` on the result, and breaks out of
the loop immediately. The agent-facing output now shows
`⚠ SEQUENCE ABORTED: goto_failed` as a banner so the next-turn
planner reads the failure as "bad URL, retry the whole interact"
rather than "53 mysterious click failures".

Covered by `tests/test_browser_interact_abort.py` — the tests exec
the runner source inline (with a stubbed Playwright import) so the
production code path itself is under test, not a reimplementation.

## Process Reward Model (`ghost_agent.prm`, 2026-04-29)

The PRM is the third inference-time learner in the pipeline (after
`router/` for request difficulty and `skills_auto/` for tool
sequences). It scores per-step `(state, action)` tuples in
microseconds against a numpy logistic regression model trained on the
same trajectory store the rest of the pipeline reads — closing the
loop between past tool-call outcomes and future plan-candidate
evaluation.

Mechanism in one paragraph: terminal `Outcome.PASSED` / `FAILED` is
back-propagated to per-step values via the AlphaZero-style γ-discount
trick (`V(step_i) = γ^(N-i-1) · terminal_value`); features are
hand-crafted (request shape + plan progress + action shape + tool
bucket + cross signals); the model is the same numpy LR shape as
`router/`, with a versioned JSON checkpoint format
(`ghost.prm.logreg.v1`). Loaded once at startup via
`PRMScorer.load(--prm-model)`, hot-swapped via `scorer.set_model(...)`
after each idle retrain pass without an agent restart.

Module layout, training pipeline, and integration details: see
[`docs/algorithms/prm.md`](algorithms/prm.md).

The PRM is **opt-in but always-attached**: `ctx.prm_scorer` is set
unconditionally in lifespan (no-op pass-through when no checkpoint is
loaded), so call sites can score `(state, action)` unconditionally
without branching on availability. MCTS engages the fast path only
when (a) `prm_scorer` is attached, (b) `has_model is True`, and (c)
the caller passes `prm_state=` — falling back to the existing
LLM-simulation path when any of those conditions miss. Existing
callers continue working unchanged; no regression to the 15/15 eval.

CLI:

```bash
# Production: load a previously-trained checkpoint at startup.
python -m src.ghost_agent.main \
    --upstream-url "http://127.0.0.1:8080" \
    --prm-model "$GHOST_HOME/system/prm/checkpoint.json"

# Bootstrap: ⚠ THIS RECIPE DOES NOT WORK AS WRITTEN (corrected §4BN).
# It used to read "omit the flag entirely; phase 2.7 will produce a
# first-ever checkpoint and hot-swap it in". Since 2026-07-27 phase 2.7
# is CONSUMER-GATED: it skips entirely unless something READS a PRM
# value — `.score()` (MCTS turn-start, needs `_MCTS_TURNSTART_ENABLED`
# AND `--deep-reason`) or `.uncertainty()` (`--frontier-selfplay` and trajectory logging).
# With neither live, no checkpoint is ever written, no matter how many
# trajectories accumulate. The sample floors below still apply ON TOP
# of that gate (≥5 trajectories, ≥20 step samples, ≥5% per class).
# To actually bootstrap one, enable a consumer:
python -m src.ghost_agent.main --frontier-selfplay

# Faster retrain cadence for development:
python -m src.ghost_agent.main --prm-train-cooldown 600
```

Coverage: `tests/test_prm_*.py` (338 tests (a MEASUREMENT — re-run the command named here rather than trusting the number) (13 modules; the enumeration below is partial — regenerate with `pytest --collect-only tests/test_prm_*.py`) across features, labels,
model, trainer, MCTS integration, biological phase, corner cases, and
adversarial fuzz/stress). Numerical hardening: NaN/inf inputs are
neutralised at `_vectorize` (inputs) and `_to_arrays` (labels) so a
single bad value can't poison the whole gradient or prediction; MCTS
defensively clamps any scorer return to [0, 1] regardless of the
scorer implementation; concurrent `score()` during `set_model()` was
exercised under 4-reader/1-swapper thread thrash. Full agent suite
remains green at **3248 passing**.

## Frontier-aware self-play (closes the PRM → self-play loop)

The PRM produces a per-step confidence signal; the trajectory store
records per-cluster coverage. Frontier-aware self-play (**default OFF** since 2026-07-09 and absent from the live launcher — §4BN R14; R5 corrected this exact claim in `docs/core/dream.html` and missed this file, so the examples below are no-ops unless you pass the flag,
`--frontier-selfplay`) combines them to choose which cluster the
biological-watchdog phase-3 self-play pass should target:

```
cluster weight  =  PRM_uncertainty(cluster)  ×  trajectory_rarity(cluster)
                   └─ 1 − 2·|p − 0.5| ─┘     └─ 1/(1 + log1p(count)) ─┘
```

Saturated clusters (per `FrontierTracker.list_saturated_clusters()`)
are excluded with weight 0. The math lives in
`core/frontier_selection.py` as pure functions; the integration is on
`FrontierTracker.pick_frontier_seed`, which mirrors the dict shape of
the legacy `pick_seed` so call sites in `core/dream.py` need no
schema branching.

**Why it matters.** The brittle-pool scoring in `pick_seed` sees
outcomes but not coverage — a cluster the agent has barely tried
looks identical to a cluster it solves first-try (both have no recent
failures). Frontier weighting surfaces the under-explored quiet ones.
That matters because the PRM is itself trained on the trajectories
self-play produces: if self-play keeps targeting the same handful of
well-trodden clusters, the PRM's opinion of the others stays stuck at
neutral, and the brittle-pool picker never gets a reason to rotate
to them. Frontier weighting breaks the loop.

**Engagement gate (strict).** `isinstance(ctx.prm_scorer, PRMScorer)
and ctx.prm_scorer.has_model and isinstance(ctx.trajectory_collector,
TrajectoryCollector)`. MagicMock-backed test contexts fail closed at
both checks, so legacy tests continue exercising the old path
unchanged. Cold-boot agents (no PRM model yet, no trajectories yet)
also fall through cleanly to `pick_seed`.

**Sanity floor.** `--frontier-uniform-sample-prob` (default 0.2)
bypasses frontier weighting on a per-tick dice roll and falls back to
the legacy `pick_seed`. Without this floor, a systematically-wrong
PRM could self-reinforce onto one cluster and starve the others of
training signal — keeping the PRM wrong about them in perpetuity.
20% uniform sampling breaks the feedback loop without losing the
benefit of frontier targeting on the other 80%.

CLI:

```bash
# ⚠ §4BN R15: these examples restated "default on" 44 lines below the
# line R14 had just corrected — the seventh recurrence of this doc-twin,
# inside the file the previous round fixed. Frontier weighting is OFF by
# default, so the first command below does NOT enable it and the
# --no-frontier-selfplay A/B is a no-op against the real default.

# Frontier weighting ON (it is OFF by default) with 20% sanity floor:
python -m src.ghost_agent.main --upstream-url "http://127.0.0.1:8080" \
    --frontier-selfplay

# A/B comparison — the DEFAULT, legacy brittle-pool pick:
python -m src.ghost_agent.main \
    --upstream-url "http://127.0.0.1:8080"

# Aggressive — drop sanity floor to 5% if the PRM is well-trained:
python -m src.ghost_agent.main \
    --upstream-url "http://127.0.0.1:8080" \
    --frontier-uniform-sample-prob 0.05
```

Coverage: `tests/test_prm_uncertainty.py` (10) +
`tests/test_frontier_selection.py` (32) +
`tests/test_frontier_pick_frontier_seed.py` (9) +
`tests/test_dream_frontier_weighted.py` (4) = 55 new tests, all
green. Existing `tests/test_dream_synthetic_curiosity.py`,
`tests/test_frontier_tracker.py`, and all `tests/test_selfplay_*.py`
continue passing — no regression to the legacy path.

End-to-end walkthrough:
[`docs/core/frontier_selection.html`](core/frontier_selection.html)
and the new section in
[`docs/algorithms/dream_cycle.html`](algorithms/dream_cycle.html).

## Meaningful self-play redesign (2026-05-17)

A post-mortem of the 2026-05-17 self-play log found that hundreds of
cycles produced **zero** lessons. The agent was solving every cycle
first-try, so:

* the compression-delta metric (tool-call count vs. prior best) stayed
  pinned at `+0.000` — no gradient for the scorer;
* the write gate (`struggled-then-won`, `new_cluster`,
  `first-failure`) never opened because every cluster had been
  seen and no cluster struggled;
* mastery (5-streak first-try wins with `delta > 0.05`) was unreachable
  because `delta` never moved;
* the reflector only fired on `outcome == FAILED`, so passing-but-
  boring cycles never reached it;
* journal mining wrote a generic `input.txt` and a lenient validator
  for every entry regardless of what the original user task was;
* PRM training only ran from the biological watchdog's 15-60 min idle
  window — but a busy self-play loop never reaches that window, so
  `PRMScorer.has_model` stayed `False` and the frontier-weighted
  picker silently fell back to the brittle pool;
* the LLM challenge generator had no incentive to produce *hard*
  challenges since it shared weights with the solver.

The redesign closes all eight gaps. Modules:

| File | Change |
|---|---|
| `core/solution_novelty.py` | **new** — AST-canonical hash + Jaccard novelty against prior winning solutions for a cluster. |
| `core/self_play_scoring.py` | multi-signal score: `passed*(1 + α·Δ + γ·novelty + δ·attempts_efficiency) − β·errors`. Defaults preserve back-compat. |
| `core/challenge_templates.py` | qualitative tier twists (`na_rows`, `negative_values`, `duplicate_ids`, `schema_drift`, …) — tier = K-combination of twists, not just N× rows. |
| `core/journal_challenges.py` | shape-aware fixtures (`input.csv` / `input.json` / `input.log` / `input.db` / `input.txt`) + shape-specific validator rubrics. |
| `core/adversarial_generator.py` | **new** — per-prompt-fingerprint solver pass-rate tracker; `suggest_bias()` injects guidance into the next challenge-gen prompt. |
| `memory/frontier.py` | per-template saturation (proposal H); ring buffer of recent winning `solution.py` sources for novelty scoring; `record_run()` now consumes `solution_source`, `template_key`, `novelty`. |
| `reflection/loop.py` | opt-in `accept_low_novelty_passes` admits self-play passes with `extra.solution_novelty < threshold` into the reflection batch. |
| `tools/memory.py` | self-play loop calls `_maybe_retrain_prm()` every 20 cycles → PRM model stays fresh without waiting for idle. ⚠ §4BN: this twin short-circuits FIRST on `prm_consumer_is_live` — on a default box nothing reads a PRM value, so it returns before training. |
| `core/dream.py` | wires all of the above: reads winning `solution.py`, computes novelty, passes it to scorer + tracker + reflector; opens write gate on `novel-shape first-try pass`; appends adversarial bias to generator prompt. |

### The new score

```
if passed:
    base = 1.0 + α·compression_delta + γ·novelty + δ·attempts_efficiency
else:
    base = 0.0
score = base − β·tool_errors
```

Defaults: α=0.4, β=0.1, γ=0.6, δ=0.3. `attempts_efficiency` is
`{1→1.0, 2→0.5, 3→0.2}`. `novelty ∈ [0, 1]` is the Jaccard distance
between the new solution's canonical AST shape bigrams and the
cluster's stored prior winning shapes (cluster cold start → 1.0;
exact AST duplicate → 0.0).

Concrete swing observed live on a fresh cold-start regex_parse cycle:
the old score reported `+1.000`; the new one reported `+1.900`
(1.0 base + 0.6 novelty + 0.3 first-try) — a real gradient that
discriminates among passes that the old score collapsed onto a single
binary outcome.

### New write-gate path

Added between `struggled-then-won` and `new-failure`:

> `passed and attempt == 0 and novelty ≥ 0.5` →
> *first-try pass with novel shape → write lesson*

A boring first-try pass with `novelty < 0.5` no longer suppresses
silently; it's reported as *"defer to reflector"* and the reflector
(when constructed with `accept_low_novelty_passes=True`) admits it
into the batch and asks *why was this boring?* — the meta-lesson
either grows the curriculum or stays the same.

### Tier twists (qualitative difficulty)

Each cluster declares an axis set in `_TWIST_AXES`. The tier-to-twist
map is:

| tier | twists picked |
|---|---|
| basic | 0 |
| intermediate | 1 |
| advanced | 2 |
| expert | 3 |

Twists are sampled deterministically from the seed so the setup and
the validator agree. `data_analysis` declares
`{na_rows, negative_values, duplicate_ids, schema_drift}` — a solver
that aced the basic shape gets four qualitatively different harder
versions to learn from, not just a 4× larger one.

### Per-template saturation (proposal H)

In addition to cluster-level saturation, each template within a
cluster gets its own outcome history. A template that earns two
consecutive first-try wins with `novelty ≤ 0.05` is marked saturated
(`saturated_at` timestamp). `list_saturated_templates()` returns
`(cluster, template)` pairs so the dreamer can rotate to a different
template within the same cluster instead of rotating the whole
cluster out.

### PRM scheduler inside the loop (proposal E)

`tools/memory._run_self_play_loop` now calls `_maybe_retrain_prm`
every 20 cycles.

> ⚠ **§4BN — recorded R18, fixed R29 (ten rounds open).** That call
> returns at `prm_consumer_is_live` **before** the trainer is reached, so
> on a default box none of the paragraph below happens: nothing trains,
> nothing hot-swaps, and the picker never engages. The trainer's own bail
> conditions are downstream of a gate that fires first. The sibling
> description 66 lines above carries this caveat; this copy did not, and
> a doc positively claiming a retrain that cannot run is §4BN's own
> defect class on the surface a scouting agent reads first.

When the gate does pass: the trainer bails out cleanly (with a logged
reason) when there aren't enough trajectories yet; on success it
hot-swaps the new `StepValueModel` into the live `PRMScorer`, and the
frontier-weighted picker (`pick_frontier_seed`) can then engage on the
next cycle instead of falling back to the brittle pool every time.

### Adversarial generator (proposal G)

`AdversarialGeneratorTracker` is keyed by a hash of the variable part
of the challenge-gen prompt (the frontier hint). It records solver
pass/fail per fingerprint, exposes `worst_fingerprints(limit)`, and
synthesises a short `suggest_bias()` block that the dreamer appends
to the system prompt for the next LLM challenge generation. Result:
the generator gets a quiet incentive to produce more challenges in
families the solver is failing on rather than rotating to easier
ones.

### Tests

New: `tests/test_self_play_meaningful.py` (44 cases) covers the
score combiner, AST novelty, twist resolver, journal shape detector,
reflector opt-in admission, adversarial tracker, write-gate inputs,
per-template saturation, and PRM scheduler safety.

Pre-existing self-play tests were updated where they pinned the old
contract:

* `tests/test_self_play_structured_lessons.py` — journal mining now
  asserts shape-appropriate fixture names instead of `input.txt`.
* `tests/test_tier_aware_templates.py` — `data_analysis` reference
  solution updated to handle the new twists; setup-script assertion
  loosened from literal `random.random() < 0.0` to
  `na_fraction = 0.0`.

Full suite: **3670 passed, 11 skipped, 0 failed.** No regressions.

## Whole-transcript post-mortem → self-defect reports (2026-06-10)

The reflection phase (2.5) raises the learning loop to "adjust how I
*act* next time" — but it only ever sees one failed turn's final
outcome + `failure_reason`, and its only durable output is a behavioural
lesson injected into prompts. That ceiling is visible in this project's
own history: the most valuable learning cycles (e.g. the June-7 triage
that produced the browser-`pageerror` capture, the fuzzy block-replace
guard, and the verifier outcome-penalty) were **tool- and control-loop
fixes** that no prompt-lesson could have produced — and they ran through
the operator by hand.

**Phase 2.5c (`--postmortem`)** automates that triage. It reads the
*whole tool-call transcript* of the worst recent FAILED runs and files a
durable, classified **defect report** into a `DefectQueue` at
`$GHOST_HOME/postmortem/defects.jsonl`:

* **behavioural** — the agent chose badly; every tool worked. Routed
  straight into `SkillMemory.learn_lesson` (same channel reflection
  uses) so it's retrieved on the next similar request, *and* logged to
  the queue with `status="routed"`.
* **configuration** — a flag/threshold let the failure mode through
  (e.g. a decay rate that let an oscillation evade the loop cap). Queued
  as a proposed config change.
* **code_defect** — a tool or the control loop is broken or blind.
  Queued, and under `--postmortem-propose-patch` an LLM-generated
  *reproducing test + unified diff* is attached. **Stored as a proposal
  only — never auto-applied.** The queue is the artifact the operator
  reviews, exactly the input that was previously assembled by hand.

### The trust anchor: a pure structural signature

Run selection and the evidence block are **LLM-free** (`compute_signature`,
`select_failed_runs`). Each transcript gets a deterministic fingerprint —
repeated-identical-error count, two-tool oscillation length, same-target
read/act-loop count, dominant tool share — blended into a `severity`
score. This does two things:

1. **No wasted calls.** A run below `--postmortem-min-severity` (default
   0.4) never costs a model call. Selection picks the top-N by severity.
2. **Grounded evidence.** The operator (and the classifier prompt) sees
   concrete facts — "the same not-found error from `read_file` recurred
   11×" — not an LLM's guess. The June-7 read-loop scores 0.66 and
   classifies as `code_defect` straight from its signature.

Dedup is by a **bucketed** signature hash: two runs that fail the same
way (11× vs 13× of the same error) collapse to one defect, so the queue
never re-files a known pathology on every idle window.

### Wiring & safety

```python
from ghost_agent.reflection import PostMortemEngine, DefectQueue

engine = PostMortemEngine(
    analyze_fn,                 # classifier LLM (wraps LLMClient.chat_completion)
    queue=DefectQueue(root),
    lesson_sink=skill_memory.learn_lesson,   # behavioural → existing channel
    patch_fn=coding_llm,        # optional; code_defect → test+diff PROPOSAL
    min_severity=0.4,
)
report = await engine.run(source=collector.iter_trajectories)
```

Same safety posture as the Reflector: never raises into the watchdog,
never mutates input trajectories, per-call timeouts, anchor-before-await
cooldown discipline (3 h default, `--postmortem-cooldown`). Read the
queue with the read-only **`postmortem`** tool (`pending` / `list` /
`show <id>` / `stats`). Tests: `tests/test_postmortem_engine.py` (34
cases — signature, selection, queue dedup/status, parsers, and the async
engine's routing / classification / patch-attachment / failure-safety).

This phase is the dual of the in-run repair loop noted under
interactive-session failures: the retry loop fixes a run *while it's
happening*; the post-mortem engine diagnoses the machinery *between
runs*. Neither auto-applies code — the post-mortem queue is a reviewed
gate by design.

### In-run companion: the no-progress loop breaker (2026-06-10)

The post-mortem engine catches this pathology *between* runs. Its live
dual closes the same loop *during* a run. The Browser-OS build surfaced
the gap: after applying a fix, the agent entered an ungrounded
verification loop — double-click the icon → screenshot → "no window" →
repeat — and narrated ~8 failed attempts with no conclusion until it hit
the turn budget. Neither existing breaker caught it:

* the error-strike counter (`execution_failure_count`) keys on tool
  **errors**; here every call *succeeded*, so it never moved;
* the cross-turn repetition breaker keys on **reasoning lexical
  overlap**; the prose varied turn to turn and slipped under the bar.

The fix is `_note_repeated_action` (companion to `_note_repeated_failure`
in `core/agent.py`), keyed by `(tool, target, result-fingerprint)` —
where `target` comes from the **same** `primary_target_from_args` the
offline signature uses. It runs in the results loop on **successful,
non-mutating** calls only:

* mutating calls (iterative `file_system` writes) are exempt, so building
  one file up over several edits is never mistaken for a loop;
* errored calls stay on the existing failure/strike path;
* a result whose content genuinely changed yields a new fingerprint, so
  real progress resets the count.

On the **first** trip (same action+target+result ≥3×) it sets
`force_final_response` and injects one directive: stop re-observing, trust
the authoritative evidence already in hand (DOM/state/file inspection),
and either report success or state plainly that the environment can't show
the result and tell the user how to verify. If it somehow loops to ≥5, it
hard-stops with a grounded `[ATTEMPT_ABORTED_NO_PROGRESS]` answer. Tests:
`tests/test_no_progress_loop_breaker.py` (15 — helpers + scenario replays
covering the Browser-OS loop, iterative-edit exemption, identical-read
loop, error exemption, and genuine-progress no-trip).

### Verifier-gate AUTO-REPAIR (2026-06-10)

The verifier gate (post-loop) catches a wrong/untested final answer and,
historically, *annotated* it — appended an auditor note, dropped
confidence, retracted any lesson the turn produced — but shipped it
anyway. The agent never got a chance to **fix** what the verifier just
flagged. Auto-repair closes that: when the verifier returns a
high-confidence `REFUTED` on the final answer (or the turn finalised on
an **unverified mutation** — an untested `file_system` write, the req_C0
"finished on a write that never ran" failure), the agent gets up to
`_MAX_VERIFIER_REPAIRS` (default **1**) extra in-loop attempts to
diagnose and correct the issue.

**Re-entry without an outer loop.** The repair reuses the existing
`for turn` loop. At the normal-success finalisation (the model produced
a final answer with no further tool calls), the verdict is computed
*there*; on a refute/unverified trigger the critique is injected as a
corrective user message and the turn loop `continue`s — the agent re-runs
with the verifier's objection in context, then re-verifies. No
~3,900-line re-indent, no method extraction of the turn loop.

Key safety properties (all tested):

* **One verifier pass on the clean path.** The verdict computed at
  finalisation is cached (`_verdict_is_fresh`) and reused by the
  post-loop gate, so a confirmed success costs exactly one verifier call
  — same as before the gate was split (`_compute_verifier_verdict` is the
  shared, side-effect-free verdict function).
* **Strictly bounded.** `repair_round` caps repairs independently of the
  turn budget; the strike-cap and all anti-loop accumulators
  (`execution_failure_count`, `repeated_failure_sigs`,
  `repeated_action_sigs`, …) are **preserved** across the repair `continue`
  (a repair turn is just another turn), so a repair can't reset the
  budgets that stop a runaway.
* **Repairs only clean successes.** Gated on
  `not tool_calls and not force_stop and execution_failure_count == 0`, so
  error / abort / terminal answers (which exit via other breaks) are never
  fed back for "repair".
* **Records once.** Trajectory recording + the `verifier_backfill` run
  once post-loop on the truly-final (possibly repaired) answer — no
  double-record.

Only thing reset on re-entry is `force_final_response` (so the repair turn
may run tools to actually fix the issue). Tests:
`tests/test_verifier_auto_repair.py` (6 — refute→repair→confirm,
critique-injection, single-pass clean cost, bounded budget,
unverified-mutation trigger, trivial-chat no-op). This is the **in-run**
counterpart to the post-mortem engine's between-runs repair; it makes the
verifier gate *act* on its verdict instead of only annotating it.

## Stage 2 hook (future work)

The trajectory log is the ingredient Stage 2 (local SFT via rejection
sampling) needs. `distill.self_consistency.pairwise_pass_fail()`
produces the (failed, succeeded) pairs; `optim.trainset.build_trainset`
consolidates them per signature. Training itself needs GPU and is
out of Stage 1 scope. The skills_auto phase now graduates verified
candidates into `auto_skills.json` (support ≥ 3, confidence ≥ 0.5,
surfaced as "PROVEN APPROACHES" + minted as `proposed` composed
macros) — and since 2026-07-05 its PASSED input actually exists in
production: the async late-verdict backfill promotes verifier-CONFIRMED
chat turns UNKNOWN→PASSED in the corpus (see
`docs/algorithms/skill_acquisition.html`, "Producer wiring").

The PRM lands as a Stage-1.5 capability: it doesn't fine-tune weights
(stays inside the no-GPU constraint) but it does close a measurable
loop — every validator-passing or user-correction-promoted trajectory
becomes a labelled training example, and the model retrains every 3
hours of idle time. Watch `pretty_log("PRM Retrain", …)` lines in the
agent log for visible improvement over time.
