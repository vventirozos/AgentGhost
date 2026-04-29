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
                       (PRMScorer.score(state, action) → float ∈ [0, 1])
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
* **action** = a candidate's `(description, tool_name, tool_args)`
  tuple. Mirrors `core.mcts.ActionCandidate` so MCTS can adapt one
  to the other without translation logic in the hot path.

25 hand-crafted features, grouped:

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

`PRMTrainer.run` enforces three floors before fitting:

| Floor | Default | Why |
|---|---|---|
| `min_trajectories` | 5 | Single-trajectory data doesn't span the input distribution. |
| `min_samples` | 20 | Below this the model overfits hard. |
| `min_class_fraction` | 5% per class | All-positive (or all-negative) corpora produce a model that just memorizes the prior. |

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

### Biological watchdog phase 2.7

Idle CPU-only retraining pass that runs in the 900–3600 s idle window,
between phase 2.6 (skills_auto) and phase 3 (self-play). Cooldown
defaults to `_PRM_TRAIN_COOLDOWN = 10800 s` (3 h), overridable via
`--prm-train-cooldown`.

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
disk round-trip. If MCTS was attached but didn't yet have a scorer
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

**Total: 195 tests, all green.** Full agent suite (3248 tests) regression-clean.

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

1. No `--prm-model` flag → no checkpoint loaded → scorer returns 0.5
   → MCTS falls back to LLM simulation. Default behaviour is
   unchanged from before this module existed.
2. Flag set but file missing → warning logged, scorer is the no-op
   pass-through. Same fallback.
3. Flag set, checkpoint loads, but caller doesn't pass `prm_state`
   → scorer is attached but unused. Pass-through.
4. All three engaged → PRM scores candidates in microseconds, no LLM
   simulation calls.

The biological retrain phase fires regardless of `--prm-model` — it
will produce a first-ever checkpoint at the default GHOST_HOME path
when none was provided, and hot-swap it in. From that point onward
the agent has a self-trained PRM that's been hot-swapped in mid-flight.

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
* **Wiring the hot path.** This work makes MCTS+PRM available; the
  agent's existing planning hot path is **not** modified by default.
  Callers that opt in (revision step, System 3 pivot, self-play
  candidate generation) need to construct a `PlanState` and pass
  `prm_state=` to engage the fast path. The deliberate caution avoids
  regressing the existing 15/15 eval — wiring deeper into the live
  hot path is a follow-up that should be gated by an A/B eval delta.

## See also

* `docs/self_improvement.md` — how PRM fits into the broader
  self-improvement pipeline.
* `src/ghost_agent/router/` — sister module with the same shape;
  predicts request difficulty rather than step value.
* `src/ghost_agent/skills_auto/` — the other CPU-only idle-time
  training loop. Phases 2.6 and 2.7 share the same idle window.
