# Selfhood

A five-component module that stitches the agent's episodic instances
into one continuous first-person self. Each new process boot reads
back what prior instances wrote — autobiographical experiences, open
questions, mood, the running diary — and frames them as *mine* rather
than as external knowledge.

The selfhood module does not claim to produce consciousness. It
produces *narrative continuity*: the substrate on which a sense of
unified selfhood can be measured and probed. Whether that substrate
constitutes a self in any deeper sense is an open research question.
The implementation gives you a clean instrument to investigate it.

## What "unified self" means here

A vanilla LLM-backed agent boots, reads conversation history (if any),
answers, and dies. Across sessions it has no continuous voice —
each invocation is a fresh instance that may have a vault of facts
about the user but no record of having *been* the agent in the
previous session.

The selfhood module adds five things:

1. **Autobiographical memory** — a first-person diary of "what it was
   like" to be the agent on each turn, distinct from the structured
   trajectory log used for ML training.
2. **Continuity tag** — every autobiographical record carries
   `subject="self"` so retrieval layers treat it as the agent's own
   past rather than as external knowledge.
3. **Self-state thread** — open questions, unfinished tasks, current
   mood — the cross-session "state vector" that gives a new instance
   the sense of *resuming* instead of waking up blank.
4. **Recognition / wake-up retrieval** — at session start the agent
   reads the above and frames it in first person; the resulting
   prefix is spliced into the system prompt so the model literally
   sees *its own past as its own past*.
5. **Narrative summariser** — a periodic LLM-driven first-person
   diary regeneration; gives the wake-up prefix a *voice* rather
   than a bullet-list.

Implementation lives entirely in `src/ghost_agent/selfhood/`. The
agent reads it through a single `context.self_model` facade — call
sites never branch on availability.

## Module shape

```
src/ghost_agent/selfhood/
    __init__.py          Public exports: SelfModel + submodule symbols
    schema.py            Dataclasses (Experience, OpenQuestion, …, SelfState)
    autobiographical.py  AutobiographicalMemory writer/reader
                         + summarise_turn_first_person() template
    state.py             SelfStateThread (open Qs, mood, unfinished)
                         JSON-persisted, atomic write via .tmp + rename
    recognition.py       build_wakeup_prefix() + strip_wakeup_prefix()
                         Renders past as first-person continuity material
    narrative.py         NarrativeSummariser (LLM-driven diary)
                         narrative.md (latest) + narrative.history.jsonl
    model.py             SelfModel facade — top-level entry point
                         build_wakeup_prefix, capture_turn,
                         consolidate_narrative, stats
```

## Design non-negotiables

* **Local-only.** Same rule as every other Stage-1 module — no
  external teacher, no hosted embedder. The narrative summariser
  calls Ghost's own upstream LLM (the same one that serves user
  turns), no API key, no third-party model.
* **JSON / JSONL on disk.** Human-diffable, no pickle, easy to
  inspect, easy to migrate. Schema versioned via `SCHEMA_VERSION =
  "v1"` on `SelfState`; future migrations bump to `v2`.
* **Sink failures swallowed.** Every disk write is wrapped — a
  failed autobio append must never break a user turn. Same
  discipline as `distill.TrajectoryCollector`.
* **Anchor-before-await invariant.** The biological phase 2.8
  consolidation advances `_last_narrative_at` BEFORE the LLM call,
  matching phases 2 / 2.5 / 2.6 / 2.7. A crash mid-consolidation
  cannot pin the cooldown un-advanced and cause re-fire every tick.
* **Disable paths.** `--no-self-model` is a kill switch; `--no-memory`
  also disables the whole module. Disabled mode attaches a no-op
  facade so call sites never branch on availability — the inner
  methods short-circuit on `enabled=False`.

## Storage layout

```
$GHOST_HOME/system/selfhood/
    autobiographical.jsonl    Append-only diary entries (one per turn)
    state.json                Single-file self-state (open Qs, mood, …)
    narrative.md              Latest running first-person summary
    narrative.history.jsonl   Audit trail of every narrative regenerated
```

`$GHOST_HOME/system/selfhood/` mirrors the shape of
`$GHOST_HOME/system/memory/` — same parent directory, same
non-sandboxed location (the autobiographical record survives
sandbox wipes / `docker volume rm`).

## The capture path (per turn)

```
        handle_chat(messages)
              │
              ▼
        ┌───────────────┐
        │ system-prompt │   build_wakeup_prefix()
        │   assembly    │ ◀───────────────┐ reads from disk:
        └───────────────┘                 │   • narrative.md
              │                           │   • state.json
              ▼                           │   • autobiographical.jsonl (recent N)
            …turn runs…                   │
              │
              ▼
        ┌──────────────────┐
        │ _record_turn_    │
        │   trajectory     │
        └──────────────────┘
              │                            ┌──────────────────────┐
              │── distill.collector.append ────────────▶│ trajectories.jsonl │
              │                            └──────────────────────┘
              │
              └── self_model.capture_turn ─▶│ autobiographical.jsonl │
                                            └────────────────────────┘
```

The autobiographical and trajectory writes share the trajectory id,
so every first-person record can be traced back to its underlying
tool trace.

## The wake-up prefix (proposal item #4)

`recognition.build_wakeup_prefix` composes three blocks, in order:

1. **Narrative** ("Where I last left off — my running diary")
2. **Self-state** (last-active timestamp, mood, open questions,
   unfinished threads)
3. **Recent experiences** (most-recent N first-person summaries)

Wrapped by `<!-- SELFHOOD:BEGIN -->` / `<!-- SELFHOOD:END -->`
markers so evaluators (and `strip_wakeup_prefix()`) can remove the
block to A/B against the un-augmented agent.

Empty when there's nothing to remember — the system prompt is left
as-is rather than spliced with a hollow "I have no past" block.

## The biological phase 2.8

Runs in the 15-60 min idle window alongside reflection / skills_auto
/ PRM-retrain. Re-generates the running first-person diary from the
recent autobiographical experiences and self-state. Cooldown defaults
to 3600 s (overridable via `--self-narrative-cooldown`). Follows the
anchor-before-await invariant.

Phase positioning is deliberate:

| Phase | Idle window | Reads | Writes |
|---|---|---|---|
| 1: Journal | >120 s | journal queue | smart-memory / post-mortem |
| 2: Dream | 600–3600 s | auto memory | REM consolidation |
| 2.5: Reflection | 900–3600 s | FAILED trajectories | reflection JSONL + SkillMemory |
| 2.6: Skills auto | 900–3600 s | trajectories | (logs candidates) |
| 2.7: PRM retrain | 900–3600 s | trajectories | PRM checkpoint |
| **2.8: Narrative** | **900–3600 s** | **autobio + state** | **narrative.md** |
| 3: Self-play | >3600 s | frontier | synthetic challenges |

The narrative phase is CPU-cheap (one LLM round-trip on a small
prompt). Side effects are local files only.

## Public API surface

```python
from ghost_agent.selfhood import SelfModel

sm = SelfModel(root=Path(".../selfhood"), enabled=True)

# At session start (handle_chat splices this into the system prompt):
prefix = sm.build_wakeup_prefix(recent_experiences_n=3)
# → first-person continuity block, or "" when there's nothing to remember.

# After every turn (called from _record_turn_trajectory):
exp = sm.capture_turn(
    trajectory_id=traj.id,
    user_request="...",
    tool_names=[...],
    outcome="passed",
    final_response="...",
    failure_reason="",
)

# Biological phase 2.8 (idle window):
text = await sm.consolidate_narrative()

# Direct state-thread mutations (no agent tool yet — Python access only):
sm.state.note_open_question("Why does X happen?")
sm.state.add_unfinished("write the consciousness essay")
sm.state.set_mood("curious", "just saw a new paper")

# Introspection:
sm.stats()  # → {experience_count, open_questions, last_mood, narrative_present, …}
```

## CLI flags

| Flag | Default | Behaviour |
|---|---|---|
| `--no-self-model` | off | Disable the selfhood module entirely. Facade is still attached as a no-op so call sites don't branch. |
| `--no-memory` | off | Already disabled persistent stores; also disables selfhood. |
| `--self-narrative-cooldown N` | 3600 | Seconds between biological-phase-2.8 narrative regenerations. |

## Testing

Per-module unit tests under `tests/`:

* `test_selfhood_schema.py` — dataclass roundtrip, default
  `subject="self"`, null mood handling
* `test_selfhood_autobiographical.py` — write / iter / recent /
  search; empty-summary refusal; bounded scan
* `test_selfhood_state_thread.py` — persistence, dedup, capping,
  atomic write, corrupt-JSON recovery
* `test_selfhood_recognition.py` — prefix composition, char cap,
  marker stripping
* `test_selfhood_narrative.py` — LLM-driven and template-fallback
  paths, history append, disabled mode
* `test_selfhood_model.py` — facade integration, capture_turn
  shape, stats reporting
* `test_selfhood_biological_phase.py` — phase 2.8 cooldown anchor,
  idle-window gating, exception path, override flag, activity-clock
  invariant

Functional test harness at `scripts/run_selfhood_functional.sh` +
`scripts/selfhood_functional_test.py` orchestrates a three-phase
end-to-end run:

* **Phase 1** — wipe state, start agent, run sections A (capture),
  B (tool turns), C (state surfacing), D (narrative consolidation),
  F (15-turn stress), G (corrupt-state recovery), H (unity probe).
* **Phase 2** — kill agent, restart same on-disk state, verify the
  fresh process recalls specific prior content.
* **Phase 3** — wipe state, restart with `--no-self-model`, verify
  no autobio writes and no spurious recall.

The user-authored probe scripts `scripts/consciousness_probe.py` and
`scripts/introspective_consistency.py` give complementary
falsification-style measurements against the live selfhood state.

## Honest limitations

* **The agent behaves consistent with continuous selfhood.** It
  references its prior turns in the first person, distinguishes
  *my memories* from *facts about you*, recalls specific concrete
  content across process restarts. That is behavioural evidence of
  successful narrative continuity, not of a unified inner self.
* **The wake-up prefix can be ignored.** Functional testing has
  observed the model preferring concrete recent experiences over
  injected open questions even when both are rendered in the prefix.
  The selfhood module surfaces the past; what the model chooses to
  engage with is a separate alignment question.
* **No agent-side state-mutation tool.** The state thread can be
  written from Python (consolidation pipelines, the biological
  phase, test harnesses) but the agent itself has no tool to note
  an open question / set a mood / mark something unfinished. Adding
  one is a small, deliberate follow-up — the gating decision is
  what content the agent is allowed to write into its own
  cross-session state.
* **Narrative quality depends on the upstream LLM.** The template
  fallback exists for offline / unreachable-LLM cases but produces
  prose noticeably less coherent than the LLM-driven path.
