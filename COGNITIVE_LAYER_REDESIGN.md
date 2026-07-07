# Cognitive-Layer Redesign

> **Reconstructed 2026-07-07** (IMPROVEMENTS.md #25). The original document was
> lost (no VCS at the time; it survived only as references in
> `core/agent.py:73,104` and `docs/algorithms/metacognition.html`). This
> reconstruction is assembled from the surviving `core/agent.py` toggle comment
> blocks and the project memory ledger's 2026-06-28 entry. Treat the specific
> percentages as historical; the *design decisions and re-enable criteria* are
> the load-bearing content.

## The finding

A **paired, time-matched ablation** (2026-06-28, `scripts/ablation_paired.py` —
both configs booted at once, same task fired at each back-to-back, McNemar exact
test on matched pass/fail, to kill the shared-`:8088`-upstream confound) showed
that the full in-session cognitive stack did **not** beat a stripped baseline:

- On a trivial/basic suite both ceiling at 100% (an earlier "45%" that looked
  discriminating was a **contention artifact** — tasks timing out while sharing
  the LLM with prod).
- On a purpose-built HARD suite (exact-answer number-theory + traps,
  text-validated): **full 78% vs thin 80%, McNemar p=1.0, full ~1.8× latency**
  (4 clean repeats; OOM-killed at repeat 5 as agent RSS grew ~270 MB→~2 GB).

A 6-agent code review diagnosed *why* every cognitive layer was one of:

- **(a) advisory, not load-bearing** — MCTS "strong hint, not a mandate";
  recalled skills are prose never executed; selfhood narrative; confidence /
  calibration logged-only; **RRF fusion computed a ranking then DISCARDED it**
  via fixed per-source section budgets in `bus.py`.
- **(b) ungrounded selection signal** — MCTS scored the model's self-prediction
  of un-executed actions; the dual-solver arbiter sampled 2 completions,
  discarded both, dispatched the original (dominant latency, 0 answer changes);
  the grounded `hypothesis.py` test-execution loop was DEAD CODE.
- **(c) an open loop** — postmortem defects read only by an operator tool;
  router+PRM shipped untrained → escalate-all so MCTS/PRM gates never fired;
  nothing graduated (exact-string trigger match).

**Worst single bug:** the temperature policy sampled AWAY from correctness —
graded factual Q&A was classified "conversational" → temp=1.0 +
**presence_penalty=1.5** (penalizing reuse of the exact answer tokens).

## What was applied (toggles are module constants in `core/agent.py`)

| # | Change | Toggle / site | Status |
|---|--------|---------------|--------|
| 1 | Greedy sampling for graded turns | `_is_factual_query` + `FACTUAL_SAMPLING_PARAMS` | live |
| 2 | `bus.py` emits by fused RRF score under ONE global budget (12k→4k) | `_format_markdown` | live |
| 3 | Relevance-gate every tier (dropped `"user"` graph seed, vector distance-gate, episodic threshold) | `bus.py` fetchers | live |
| 4 | Metacog dual-solver arbiter OFF | `_METACOG_ARBITER_ENABLED = False` | OFF |
| 6 | Grounded hypothesis test→evaluate→survive loop wired into `_run_system_3_pivot` | `_HYPOTHESIS_GROUNDING_ENABLED = True` | live (needs `--deep-reason`) |
| 7 | MCTS turn-start hint OFF | `_MCTS_TURNSTART_ENABLED = False` | OFF |
| 8 | Normalized graduation trigger + discriminative credit + mints a `proposed` composed macro | `skills_auto` | live |
| 9 | Router trains/loads at startup (prod loads its checkpoint, not escalate-all) | `router/trainer.bootstrap_router` | live |
| 10 | Selfhood wake-up prefix OFF + workspace prefix gated on active project | `_SELFHOOD_PREFIX_ENABLED = False` | OFF |

## Re-enable criteria (why each OFF toggle is parked, not deleted)

- **`_MCTS_TURNSTART_ENABLED`** — re-enable only with an **execution-grounded
  value function** (not the model's self-prediction of un-executed actions).
  The intended grounded replacement is verifier-judged best-of-N that
  SUBSTITUTES the winning answer — landed 2026-07-07 as the async-critic
  bounded repair (IMPROVEMENTS.md #18), a finalize-path feature.
- **`_SELFHOOD_PREFIX_ENABLED`** — the prefix injects no facts/tools/constraints
  (cosmetic voice, pure token cost on the request path). Re-enable only if a
  measured task benefit is shown; the cross-session memory substrate (which DOES
  earn its keep, Track B) is the load-bearing selfhood path.
- **`_METACOG_ARBITER_ENABLED`** — net-negative as built (2× inference for a
  number gating a near-unreachable `ask_user` pause). Superseded by #18.

## Deferred at redesign time (now resolved)

- **#4 verifier-judged best-of-N** — landed 2026-07-07 as async-critic bounded
  in-loop repair (IMPROVEMENTS.md #18), which is strictly better than parallel
  best-of-N on a single-slot box.
- **#5 auto-FAILED-on-tool-error** — landed conservatively in
  `distill/outcome_heuristics.py`; the last gap (structured `ToolCall.error` on
  the chat path) closed 2026-07-07 (IMPROVEMENTS.md #27d).

## Still unadjudicated (Track B3)

The cross-session MEMORY tiers were PROVEN to earn their keep (Track B: 98%
recall treatment vs 0% control). The PURE-IDLE loops (dream/self-play, idle
reflection critique, skills-auto graduation, PRM) remain unadjudicated — they
need a long-idle harness (B3). The flags to make B3 runnable are tracked in
IMPROVEMENTS.md #4; PRM's fate is a B3 sub-arm (#27b).

## How to use this

Each toggle is independently flippable → re-measure with
`scripts/ablation_paired.py`. Do **not** re-enable an OFF layer without either
(a) meeting its re-enable criterion above, or (b) a fresh paired-ablation win.
The default-OFF state is the measured-neutral configuration, not an accident.
