# Process Reward Model (PRM)

A learned per-step value estimator. Scores `(state, candidate_action)`
tuples in microseconds without executing the action — used by
`core.mcts.MCTSReasoner` as a fast path so plan candidates are scored
against the trained model instead of paying a worker-LLM simulation
call per candidate.

The PRM closes the largest open gap in Ghost Agent's reasoning loop:
the planner is otherwise myopic. Single-shot LLM planning picks step 1
without simulating whether step 2 has good options. With a PRM-guided
tree search the agent can imagine `K` candidate first actions, score
them all, expand the best `M`, and only execute the highest-scoring
leaf path — the same technique responsible for the o1 / R1 / DeepSeek
reasoning gains in 2024-2025, adapted to a tool-using agent.

## Module shape

```
src/ghost_agent/prm/
    __init__.py        Public exports: PRMScorer, PRMTrainer, …
    features.py        Per-step feature extraction
                       (PlanState, ActionFeatures → FeatureVector)
    labels.py          MC value backprop from terminal Outcome
                       (StepLabelSpec, derive_step_labels, iter_step_samples)
    model.py           numpy-only logistic regression
                       (StepValueModel, JSON-persisted, schema-versioned)
    scorer.py          Production-facing wrapper
                       (PRMScorer.score(state, action) → float ∈ [0, 1];
                        PRMScorer.uncertainty(state, action) → float ∈ [0, 1])
    trainer.py         Pipeline: trajectories → samples → fit → save
                       (PRMTrainer with bail-on-bad-data semantics)
```

## Design non-negotiables

* **Local-only.** Pure-numpy logistic regression. No outbound traffic
  at feature time, no embedding service, no hosted scorer. Same rule
  as `router/` — and the implementation deliberately mirrors
  `router/model.py` so future readers don't have to learn two
  conventions.
* **JSON-persisted (not pickle).** The schema is human-diffable, safe
  to load (no code-execution risk), and the format is versioned
  (`ghost.prm.logreg.v1`). Future model swaps (MLP, EBM) land at
  `ghost.prm.<kind>.v1`.
* **Fail-safe.** A scorer with no trained model returns a neutral 0.5
  for every candidate. Callers can score unconditionally; when no
  model is loaded, MCTS effectively falls back to its existing LLM
  simulation path because every candidate ties.
* **Frozen feature ordering.** `PRM_FEATURE_NAMES` is the contract.
  New features must be APPENDED, not inserted. Saved checkpoints
  record the names; `load` raises on schema drift instead of
  silently mis-aligning weights against a different feature layout.

## What the PRM scores

A single `(state, action)` tuple where:

* **state** = the prefix-state immediately before the candidate action
  fires (request shape, plan progress, tools used so far this turn,
  tools that errored so far). Reconstructed at training time so the
  PRM only sees what the agent actually knew at decision time —
  leaking later steps would let it post-hoc infer the answer.

  > **Retrain note (May 2026).** A label-leakage bug was fixed: the
  > prefix-state's `pending_count`/`plan_depth` were being derived from
  > the *completed* trajectory (i.e. the future), which the MC label is
  > monotone in — so the PRM could learn the label almost directly. They
  > are now pinned to neutral inference constants. **PRM checkpoints
  > trained before this fix must be retrained.** See
  > [Audit &amp; hardening](../audit_fixes.html).

  > **Retrain note (2026-07-13).** The same pinning now covers the whole
  > plan-progress block: `steps_so_far`, `failures_so_far`,
  > `tools_used_this_turn`, `tools_failed_this_turn` are fixed to the
  > turn-start constants (0 / 0 / () / ()) because every live scoring
  > site scores at turn start — mid-turn training variance was pure
  > train↔serve skew (and `steps_so_far` mildly label-leaking: it equals
  > the step index the MC label is monotone in). Checkpoints trained
  > before this refresh automatically on the next idle retrain.
* **action** = a candidate's `(description, tool_name, tool_args)`
  tuple. Mirrors `core.mcts.ActionCandidate` so MCTS can adapt one
  to the other without translation logic in the hot path.

26 hand-crafted features, grouped:

| Group | Features | Purpose |
|---|---|---|
| Request shape (8) | length, code-fence, URL, imperative count, jargon count, question-words ratio, has `?` | Captures task type signals already proven useful by the router. |
| Plan progress (5) | `steps_so_far`, `failures_so_far`, `pending_count`, `plan_depth`, `has_any_failure` | The agent's current trajectory state. |
| Action shape (5) | description length, args count, args total length, has-URL-in-args, has-filepath-in-args | What the candidate actually proposes. |
| Tool bucket (5) | `is_heavyweight` / `is_lightweight` / `is_external` / `is_memory` / `is_unknown` | Coarse one-hot over tool category. Buckets (not per-tool one-hots) so adding a new tool doesn't invalidate every checkpoint. |
| Cross (2) | `tool_already_used_this_turn`, `tool_failed_this_turn` | Catches refire on already-failing tools. |

Adding a feature: append to `PRM_FEATURE_NAMES`, compute it in
`extract_step_features` (the runtime check fails loud if you forget),
retrain. Old checkpoints reject the load to prevent silent misalignment.

## Label derivation: Monte Carlo value backprop

Same technique AlphaZero uses to credit-assign through a winning
rollout. Given a trajectory with `N` tool calls and terminal outcome:

```
o == PASSED  →  terminal value = 1.0
o == FAILED  →  terminal value = 0.0
o == UNKNOWN →  trajectory skipped (no useful gradient)

For each step i (0-indexed):
    V(step_i) = γ^(N-i-1) · terminal_value
```

With γ = 0.9 and N = 4 → values `[0.729, 0.81, 0.9, 1.0]`. The step
right before the win gets full credit; earlier steps get exponentially
less. In a FAILED trajectory all steps get 0 (every step in a losing
rollout is a counterexample). UNKNOWN trajectories are dropped, not
guessed at — sparse labels beat noisy ones for a fail-safe scorer.

The continuous value is what `PRMTrainer` consumes by default
(`use_continuous_labels=True`). Set `False` to train on
threshold-binarized labels instead — simpler, but loses the
how-close-to-success encoding.

## Training pipeline

```
$GHOST_HOME/system/trajectories/YYYY-MM-DD/session-*.jsonl
                                  ↓
              TrajectoryCollector.iter_trajectories()
                                  ↓
                  iter_step_samples(spec=StepLabelSpec())
                                  ↓
                       List[StepSample]
                       (state, action, value, binary, …)
                                  ↓
              extract_step_features(state, action) per sample
                                  ↓
                         StepValueModel.fit
                                  ↓
                      save → JSON checkpoint
                                  ↓
                  PRMScorer.load(path) → live scoring
```

`PRMTrainer.run` enforces size + viability floors before fitting:

| Floor | Default | Why |
|---|---|---|
| `min_trajectories` | 5 | Single-trajectory data doesn't span the input distribution. |
| `min_samples` | 20 | Below this the model overfits hard. |
| `min_class_fraction` | 5% per class | **Binary mode only** (`use_continuous_labels=False`): a one-sided gradient runs the bias to ±∞. |
| `min_label_std` | 0.02 | **Continuous mode only:** near-constant soft targets carry no gradient to fit. |

**The viability floor is matched to the label mode (fixed 2026-07-07).**
The trainer's default is `use_continuous_labels=True` — it fits the
discount-weighted soft values, not the 0/1 threshold. But the gate used
to bail on the **binary** class balance (`min_class_fraction` on the 0.5
threshold view), so a perfectly trainable set whose soft values all sit
on one side of 0.5 (e.g. a mostly-failing corpus with a few high-value
success anchors → ~3% binary-positive) was wrongly rejected as "class
imbalance." Now: continuous mode requires **both regimes represented**
(≥1 success-side and ≥1 failure-side sample — so an all-PASSED or
all-FAILED corpus still bails) **plus** a label-variance floor
(`min_label_std`), and it does *not* re-impose the fraction floor;
binary mode keeps the original per-class fraction floor.

**Train↔serve feature skew — RESOLVED 2026-07-13 (check kept as
tripwire).** Several turn-progress features (`plan_steps_so_far_log1p`,
`plan_failures_so_far_log1p`, `plan_has_any_failure`,
`tool_already_used_this_turn`, `tool_failed_this_turn` —
`SERVE_TURN_START_INERT_FEATURES`) are **always 0 at the live scoring
sites**, which fire at TURN START (agent.py MCTS lookahead,
`frontier_selection.representative_state`: no step run, no tool used
yet). They used to vary across training samples (drawn mid-turn), so
the fit leaned on signal the deployed model could never see — surfaced
as the "serve-inert features vary in training" warning on every idle
retrain. The training-signal redesign landed: `_build_state_for_step`
now pins the **entire plan-progress block** to the turn-start constants
(steps=0, failures=0, no tools used/failed, pending=1, depth=1), so
training and serving see identical values and only the request text and
candidate action carry gradient. The skew check in `run` remains as a
regression **tripwire**: it fires only if mid-turn variance is
reintroduced without moving the scoring sites in lockstep
(`TrainerReport.feature_skew_warning`, mirrored into `summary()` with a
⚠). If a future call site needs mid-turn scoring, un-pin the fields and
make that site pass its real progress state — training and serving must
move together. Tests: `tests/test_prm_binary_floor_and_skew.py` (both
directions), `tests/test_prm_labels.py` (pin contract).

When ANY floor isn't met, `run` returns a `TrainerReport` with
`fit_attempted=False` and a human-readable `bail_reason`. **No
checkpoint is written.** That's deliberate: a confidently-wrong scorer
poisons every plan it scores until the next retrain pass overwrites
it. Returning empty-handed forces the watchdog to retry next cycle
with whatever fresh data has accumulated.

## Wiring into the agent

### CLI flags (`main.py::parse_args`)

```
--prm-model PATH            Path to a persisted PRM JSON checkpoint.
                            When set, the scorer loads on startup and
                            plugs into the MCTS reasoner. When unset,
                            the context still gets a no-op scorer
                            (returns 0.5 for every candidate) so
                            callers can score unconditionally.
--prm-train-cooldown SECS   Seconds between idle-time PRM retrain
                            passes (biological phase 2.7).
                            Default 10800 (3 hours).
```

### Lifespan (`main.py::lifespan`)

```python
context.prm_scorer = PRMScorer()
if args.prm_model and Path(args.prm_model).exists():
    context.prm_scorer = PRMScorer.load(Path(args.prm_model))

# When MCTS is enabled and PRM has a trained model, plug it in.
if context.mcts_reasoner is not None and context.prm_scorer.has_model:
    context.mcts_reasoner.prm_scorer = context.prm_scorer
```

A scorer is always attached to the context — the no-op fallback
returns 0.5, which lets all call sites do
`ctx.prm_scorer.score(state, action)` without branching on availability.

### MCTS integration (`core/mcts.py::MCTSReasoner`)

```python
mcts = MCTSReasoner(
    llm_client=llm,
    max_candidates=3,
    prm_scorer=ctx.prm_scorer,   # optional
)

winner = await mcts.select_best_action(
    task=...,
    plan_state=...,
    available_tools=[...],
    prm_state=PlanState(           # NEW: opt-in PRM scoring
        user_request=user_text,
        steps_so_far=n_done,
        ...,
    ),
)
```

The fast path engages when **all three** conditions hold:

1. `prm_scorer` was passed to `MCTSReasoner.__init__` (or assigned to
   `mcts.prm_scorer` after construction).
2. The scorer's `has_model` is `True` (a checkpoint actually loaded).
3. The caller passed `prm_state` to `select_best_action`.

Fail any of those three and the legacy LLM-simulation path runs —
existing callers continue working unchanged. Backwards compatibility
is verified by `tests/test_mcts.py` (the original 12 tests) and
`tests/test_deep_reason_wiring.py` (still pass after the integration).

### Frontier-aware self-play (consumer of `uncertainty()`)

`PRMScorer.uncertainty(state, action)` is the boundary-distance proxy
`1 − 2·|p − 0.5|`. Scores at the rails (0 or 1) map to 0.0
(maximally confident); scores at p = 0.5 map to 1.0 (maximally
unsure). When the scorer has no loaded model the underlying `score`
returns its neutral 0.5, which by this metric IS maximum uncertainty —
semantically correct: "we have no opinion" and "we are most unsure"
are the same posture for an untrained logistic regression. Never
raises; on internal error returns 1.0 to bias toward exploration
rather than silently dropping the cluster.

The biological-watchdog phase-3 self-play picker
(`core/dream.py::synthetic_self_play`) consumes this via
`core/frontier_selection.py::compute_cluster_uncertainty` — see
`docs/core/frontier_selection.html` for the weighting math and
`docs/algorithms/dream_cycle.html` for the wire-up. Engagement is
strict-typed: the scorer must be an actual `PRMScorer` instance with
`has_model=True` (not a MagicMock test attribute) AND the trajectory
collector must be an actual `TrajectoryCollector`, otherwise the
picker falls through to the legacy brittle-pool path. Covered by
`tests/test_prm_uncertainty.py` (10 cases) plus the frontier and
dream-integration test files.

### Biological watchdog phase 2.7

Idle CPU-only retraining pass that runs in the 900–3600 s idle window,
between phase 2.6 (skills_auto) and phase 3 (self-play). Cooldown
defaults to `_PRM_TRAIN_COOLDOWN = 10800 s` (3 h), overridable via
`--prm-train-cooldown`.

**Consumer gate (2026-07-27, the part this page omitted).** Before any
of the below, the phase asks whether *anything reads a PRM value*:

```python
_prm_consumer_live = prm_consumer_is_live(ctx)     # core/agent.py

def prm_consumer_is_live(ctx) -> bool:
    score_live = bool(_MCTS_TURNSTART_ENABLED
                      and getattr(ctx, "mcts_reasoner", None) is not None)
    uncertainty_live = bool(
        getattr(getattr(ctx, "args", None),
                "frontier_selfplay", False) is True
        # …AND a real collector: the frontier picker's call site
        # requires one, so --no-trajectories makes this leg dead.
        and getattr(ctx, "trajectory_collector", None) is not None)
    return bool(score_live or uncertainty_live)
```

⚠ This page previously published the predicate as
`bool(_MCTS_TURNSTART_ENABLED or ctx.args.frontier_selfplay is True)`.
That was the code, and it was **wrong**: `.score()`'s call site also
requires a live `ctx.mcts_reasoner` (i.e. `--deep-reason`), so with the
constant flipped and `--deep-reason` off the phase trained a model
nothing could read — the very defect the gate exists to prevent. Fixed
in §4BN R3; the predicate now lives in ONE place and both gates (phase
2.7 and the twin in `tools/memory.py`) call it. The skip message is
derived by `prm_consumer_why_no_reader(ctx)` so it names the conjunct
that is actually missing rather than a hardcoded one.

If not, it **skips and logs why**. It had previously run 41 retrains in
one ledger window while logging "value model refit" — learning-shaped
noise for a model nothing consumed. Two things follow, and both bit
later work:

- The cooldown is **not** gated on `--prm-model` (the CLI help said it
  was; corrected in §4BN). It is gated on consumers.
- `--prm-online-update` is a **PRODUCER**, not a third consumer, and is
  deliberately excluded from this predicate. Widening it to count would
  reinstate exactly the 41-wasted-retrains defect, with a grinder
  attached — §4BM registered that widening, §4BN retracted it before
  any code was written. The twin gate in `tools/memory.py` carries the
  same predicate and the same exclusion.

```
─── 900s idle ───────────────────────── 3600s idle ───
        ┌─ phase 2 (REM dream)
        ├─ phase 2.5 (reflection)
        ├─ phase 2.6 (skills_auto)
        └─ phase 2.7 (PRM retrain) ────  ← NEW
                  ↓
                  ↓                           ↓
              phase 3 (self-play, > 3600s idle)
```

The phase respects the same cooldown-anchor invariants as every other
phase: anchor advances BEFORE the await AND in `finally`, so an
exception mid-fit doesn't leave the cooldown un-reset (which would
cause the failing fit to refire every 60 s for the rest of the idle
window). The activity clock (`ctx.last_activity_time`) is NOT touched
— that's the user's clock, and resetting it would starve phase 3.

After a successful fit, the trainer's freshly-trained model is
hot-swapped into the live `ctx.prm_scorer` via `set_model(...)`. The
**very next** plan score uses the new weights — no agent restart, no
disk round-trip. (On the live box there is no next plan score: `.score()`
is reachable only from MCTS turn-start, which `_MCTS_TURNSTART_ENABLED`
holds off. The hot-swap machinery is correct and dormant.) If MCTS was attached but didn't yet have a scorer
plugged in (first-ever fit case), the phase bridges them too.

## Tests

| File | Coverage |
|---|---|
| `tests/test_prm_features.py` | 32 cases — vector shape, deterministic ordering, request features, plan features, action features, tool buckets, cross features, edge cases (huge inputs, missing keys). |
| `tests/test_prm_labels.py` | 24 cases — MC value math (γ-discount), terminal-outcome handling, min_steps floor, include_failed toggle, prefix-only state reconstruction, lazy iteration. |
| `tests/test_prm_model.py` | 20 cases — fit/predict/save/load, soft-label support, sigmoid clipping, schema-version check, feature-drift rejection, atomic-replace save. |
| `tests/test_prm_trainer.py` | 11 cases — successful fit, in-memory mode, predicted-value bounds, checkpoint identity, all three bail reasons, no-checkpoint-on-bail. |
| `tests/test_prm_mcts_integration.py` | 7 cases — fast-path activation, fast-path skipped when conditions miss, score bounds, provenance string, exception isolation, hot-swap pickup. |
| `tests/test_prm_biological_phase.py` | 12 cases — phase fires when wired, gating, cooldown anchor, exception advance, user-supplied cooldown, no-swap on bail, MCTS auto-plugin. |
| `tests/test_prm_corner_cases.py` | 59 cases — all-UNKNOWN / all-PASSED / all-FAILED corpora, malformed tool_args, NaN/inf labels and inputs, NaN discount, wrong-length vectors, corrupted JSON, missing fields, atomic save, scorer edge inputs, MCTS scorer-returns-NaN/out-of-range, trainer repeat-run, concurrent score during set_model swap, 5K-sample stress, 1K-candidate batch scoring, corrections-sidecar overlay changing labels, phase 2.7 with unwritable parent, anchor-advance on trainer mid-iteration crash. |
| `tests/test_prm_adversarial.py` | 30 cases — randomised fuzz inputs to feature extraction (10 seeds × 200 inputs each), random balanced-corpus fuzz training (5 seeds), 10K-sample fit under 60 s, 5K-candidate batch scoring, tool args with null bytes / control chars / 50-deep nesting / circular references / injection-shaped strings, 1000-iteration set_model thrash, schema migration rejections (legacy v0, partial feature names), feature-tuple immutability, dtype consistency (FeatureVector ≡ ndarray ≡ list ≡ tuple). |
| `tests/test_prm_uncertainty.py` | 10 cases — `PRMScorer.uncertainty(state, action)` contract: untrained → 1.0; trained at p=0 / p=1 → 0.0; p=0.5 → 1.0; symmetric quarter-points → 0.5; exception isolation; NaN robustness. Consumed by `core/frontier_selection.py`. |

**Total: 338 tests across tests/test_prm_*.py, all green** (a MEASUREMENT, not a constant — re-run the command in this sentence rather than trusting the number; it has been stale in three consecutive rounds) (the per-module list above is partial and drifts; regenerate with `pytest --collect-only tests/test_prm_*.py`). The full agent suite is regression-clean; its count has grown well past the 3248 recorded here.

## Numerical hardening invariants (proven by test_prm_corner_cases.py)

These properties are now contractually enforced and exercised by tests:

| Invariant | Where enforced |
|---|---|
| NaN feature inputs do NOT propagate into predictions | `model._vectorize` calls `np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)` |
| NaN labels do NOT poison fit gradients | `model._to_arrays` calls `np.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0)` then clips |
| NaN discount factor falls back to default 0.9 | `labels.derive_step_labels` checks `math.isfinite(gamma)` and clamps |
| MCTS clamps any scorer return to [0, 1], not just `PRMScorer` | `mcts._clamp_unit_score` mirrors `prm.scorer._clamp_unit` for duck-typed scorers |
| Concurrent `score()` during `set_model()` is safe | Tested with 4-reader + 1-swapper threads × 0.5 s of thrash; zero errors |
| Trainer that bails preserves the previously-fit model | `PRMTrainer.run` only assigns `self.model` after `fit_succeeded` |
| Save under fault doesn't lose the in-memory model | `trainer.model` is set before save attempt |
| 10K-sample fit completes in < 60 s on commodity hardware | `test_10k_sample_fit_under_60s` |
| 5K-candidate batch scoring completes in < 10 s | `test_5k_candidate_scoring_batch` |

## When the PRM activates

The PRM is **opt-in** at multiple levels:

1. No `--prm-model` flag → ⚠ **§4BN R14: stale.** Boot falls back to the DEFAULT checkpoint path (`$GHOST_HOME/system/prm/checkpoint.json`) and loads it when present, so "unset" does not imply "no model". Only when neither exists does the scorer return 0.5
   → MCTS falls back to LLM simulation. Default behaviour is
   unchanged from before this module existed.
2. Flag set but file missing → warning logged, scorer is the no-op
   pass-through. Same fallback.
3. Flag set, checkpoint loads, but caller doesn't pass `prm_state`
   → scorer is attached but unused. Pass-through.
4. All three engaged → PRM scores candidates in microseconds, no LLM
   simulation calls.

⚠ **Not on the live box.** The sentence that used to sit here — "the
biological retrain phase fires regardless of `--prm-model` … from that
point onward the agent has a self-trained PRM" — has been false since
2026-07-27, when the phase gained a **consumer gate** (see phase 2.7
below). With `_MCTS_TURNSTART_ENABLED` False and `--frontier-selfplay`
unset, the retrain SKIPS, no checkpoint is ever written, and none of
the four "all engaged" states above is reachable without operator
action. This stale paragraph is what a scouting pass read in §4BM to
conclude the retrain predicate was too narrow; the resulting plan was
retracted in §4BN. Verify against the code (`agent.py` phase 2.7), not
against this page's history.

## Online updates (low-latency correction learning)

The batch retrain (phase 2.7) runs at most every 3 hours of idle time, so
a user correction at turn N can't influence the PRM until hours later.
`StepValueModel.partial_fit(X, y, *, lr, steps)` closes that gap with a
small in-place SGD step on the existing weights (the online counterpart
to batch `fit`; same numpy, no GPU). It requires an already-fitted model
— online steps *refine* the batch model, they don't bootstrap one.

`PRMScorer.online_update(new_X, new_y, *, holdout_X, holdout_y)` is the
guarded entry point:

* **clone, don't mutate** — the step is applied to a `clone()`; the live
  scoring model is replaced atomically via `set_model` only if the guard
  passes, so a concurrent `score` never sees a torn update;
* **holdout guard** — the clone is committed only if its BCE on a holdout
  of recent trajectories does **not** worsen (catastrophic-forgetting
  guard via `bce_loss`); ⚠ **§4BN R22/R23: the no-holdout path is no longer reachable from the
  wired call site.** An empty holdout makes `online_update` set
  `base_loss=None` and commit unconditionally — so
  `_run_prm_online_update` now SKIPS the step when the holdout is below
  `_PRM_ONLINE_MIN_HOLDOUT` (5 samples) and logs a `PRM Online Skipped`
  WARNING. Measured on the live store: median 74 holdout samples, 0%
  below the floor, so this does not make the flag inert in practice. The
  scorer's own bounded-step reasoning below still applies to any other
  caller;
* **no bootstrap** — returns `False` when no model is loaded.

Wired (opt-in, `--prm-online-update`) at the user-correction promotion
path in `core/agent.py`: when a turn is promoted to FAILED by a user
correction, its step samples become the new (negative) sample and a slice
of recent trajectories the holdout — applied fire-and-forget in a worker
thread so the turn never blocks. The same `partial_fit` / `clone` /
`bce_loss` trio is mirrored on `router.model.ComplexityClassifier`.
Covered by `tests/test_prm_online_update.py`.

**Three ways this flag can be inert:** (a) no model to refine; (b) nothing that would read a refinement; (c) trajectory logging off, so the user-correction path returns before the dispatch and no update is ever ATTEMPTED — independently of (a) and (b). Boot names every reason that applies. A FOURTH limitation is architectural rather than config-dependent, so boot cannot detect it: the step is dispatched only from an inline user correction, never from a `/api/feedback` label (§4BN R25). The
"no bootstrap" bullet above is a hard floor, and the consumer gate on
phase 2.7 removes the usual way of getting past it:

| state | what happens |
|---|---|
| flag set, no model loaded | every update returns `False` — nothing refines |
| flag set, model loaded, no `.score()`/`.uncertainty()` consumer live | updates commit, and nothing ever reads the refined weights (they also die at process exit — `online_update` commits via `set_model`, it does not persist to disk) |
| flag set, model loaded, a consumer live | working as designed |

`main.prm_online_update_inertness(flag_set, has_model, frontier_selfplay,
score_module_gate, score_reasoner_present, trajectory_logging,
deep_reason)` is a pure
function that returns the operator-facing WARNING when the flag cannot
work, and `None` when it can. Boot calls it through the single
`log_prm_boot_warnings(context)` hop and logs it under **PRM Online
Update**.

**Third boot warning: `PRM Consumer Inert` (§4BN R9/R10/R11).** Distinct
from the two above, which are about `--prm-model` and
`--prm-online-update`. This one fires when `--frontier-selfplay` is set
but its consumer cannot run — the frontier picker needs BOTH a real
`TrajectoryCollector` AND a fitted model (`has_model`), and the latter is
deliberately excluded from `prm_consumer_is_live` because including it
would deadlock the retrain. Before it existed, that box was silent
everywhere: boot said nothing, phase 2.7 logged at debug, the twin logged
at debug, and `dream.py` logged nothing at all because the branch is
simply not taken. Its message names whichever conjunct is missing, and
says so without claiming the whole box is inert when the other leg is
live.

> ⚠ **This signature has now been stale FOUR times in this file** (4-arg →
> 5-arg → 6-arg → 7-arg), each time while the prose beside it claimed the function
> "reads the WHOLE gate". Since it is advertised as importable, a second
> caller following a stale signature silently omits a conjunct and
> re-creates the defect. If you change the parameters, change this line in
> the same edit.

It reads the `.score()` gate at runtime rather than assuming it, and reads
**both legs** of the consumer question — but NOT every conjunct of either
call site, and the difference matters. `.score()` is live only when
`_MCTS_TURNSTART_ENABLED` **and** `context.mcts_reasoner is not None`
(i.e. `--deep-reason`).

⚠ **The census here is deliberately incomplete, and "reads the WHOLE gate"
was false when this page said it.** The real call sites carry conjuncts
this gate does not read: `core/dream.py` additionally requires
`isinstance(prm_scorer, PRMScorer)`, `has_model`, and an enclosing
`isinstance(frontier_tracker, FrontierTracker)`; `core/agent.py`'s
turn-start block additionally requires `_is_hard`, a non-empty user turn,
and a non-trivial chat.

⚠ **§4BN R15: the exclusion list below describes `prm_consumer_is_live`,
NOT the function documented above.** Since R13 MAJOR-4, `has_model` is
parameter #2 of `prm_online_update_inertness` and IS read there — the
retrain GATE excludes it (a boot message and a training gate ask
different questions), and this block sat under the signature of the
function that reads it. `has_model` is excluded from the GATE on
purpose — including it
would deadlock the retrain (no model ⇒ never train ⇒ never a model). The
others are excluded because they cannot be False in the configurations
this gate is asked about. Recorded rather than quietly omitted: this is
the page whose staleness is the documented cause of §4BM's bad
registration, so a completeness claim here is expensive.

The first version hardcoded "module-gated off"
into the message; the second read only the module constant, which is
necessary but not sufficient — both would have lied on a box where the
flag had just started working, or gone silent on one where it hadn't
(`tests/test_prm_online_update_loudness.py`).

## Honest tradeoffs

* **Corpus size.** Logistic regression is the right shape for a small
  trajectory store. When the store grows past a few thousand
  validator-passing samples, the model becomes the bottleneck. Schema
  is versioned; future MLP / small transformer / EBM swaps land
  alongside as `ghost.prm.<kind>.v1` without touching the call sites.
* **Reward hacking.** A model that scores well on the PRM but executes
  poorly is the classic failure mode. The verifier remains the
  terminal arbiter — when it disagrees with a PRM-favoured plan, the
  verifier wins. Operators should monitor PRM-vs-verifier disagreement
  rate as a calibration signal.
* **Discount factor coupling.** γ = 0.9 is well-matched to typical
  agent turns (1–4 steps). Pushing toward γ = 1.0 makes the model
  treat all steps in a passed trajectory as equally valuable —
  helpful when most failures are diffuse, harmful when they're
  concentrated near the end. Tunable via `StepLabelSpec.discount_factor`.
* **Wiring the hot path.** The deep-reason MCTS lookahead in
  `core/agent.py::handle_chat` now **constructs a `PlanState` at turn
  start and passes `prm_state=`** to `select_best_action`, so a trained
  PRM scores live candidates in microseconds instead of the lookahead
  always paying 3–4 worker-LLM simulation round-trips. The turn-start
  state pins `pending_count`/`plan_depth` to the same neutral constants
  (`1/1`) used by `prm.labels._build_state_for_step` and
  `frontier_selection.representative_state`, so there is zero train/serve
  skew (asserted by `tests/test_prm_mcts_live_wiring.py`). When no
  trained PRM is loaded, the MCTS gate falls back to LLM simulation
  automatically — the fast path engages only once a checkpoint exists.
  ⚠ There are **no other callers** (&sect;4BN R10): `select_best_action` has
exactly one call site in `src/`, the MCTS turn-start hint. This sentence
used to list a revision step, a System 3 pivot and self-play candidate
generation as following "the same pattern" — overstating the `.score()`
census on the page whose staleness caused §4BM's bad registration, and
contradicting the retraction's own load-bearing fact.

## See also

* `docs/self_improvement.md` — how PRM fits into the broader
  self-improvement pipeline.
* `src/ghost_agent/router/` — sister module with the same shape;
  predicts request difficulty rather than step value.
* `src/ghost_agent/skills_auto/` — the other CPU-only idle-time
  training loop. Phases 2.6 and 2.7 share the same idle window.
