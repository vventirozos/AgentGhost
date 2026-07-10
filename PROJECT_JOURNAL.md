# Ghost Agent — Project Journal

The single source of truth for the agent's hardening history, architecture
decisions, operational conventions, and open work. **Supersedes and replaces**
`BUGHUNT.md`, `BUGHUNT_FUNCTIONAL.md`, `COGNITIVE_LAYER_REDESIGN.md`, and
`IMPROVEMENTS.md` (all folded in here 2026-07-07).

Structure:
- **1. Current state** — one-screen summary.
- **2. Operational reference** — how to run/restart/test the live agent (load-bearing).
- **3. Cognitive-layer redesign** — the toggles + re-enable criteria (cited by `core/agent.py`).
- **4. WHAT REMAINS TO DO** — the consolidated open work (start here for the next session).
- **5. Completed ledgers** — the 27-item improvement board + the two bug-hunt unit tables.
- **6. Session history** — chronological log of notable fixes.

---

## 1. Current state (2026-07-07)

- **Static bug hunt (source review): COMPLETE.** All 28 units CLEAR (`utils` → `core/agent.py`
  → `scripts`), 2026-07-03/04. Every subsystem reviewed for concrete-failure bugs; confirmed
  bugs fixed with regression tests + HTML docs. Residual/uncertain findings live in §4.
- **Functional bug hunt (live agent on :8000): COMPLETE.** All 14 subsystems CLEAR, 2026-07-04.
  Real `/api/chat` requests drove each subsystem; ~10 real fixes shipped (turn-loop parse,
  insert_fact hang, native tool-call repair ×3, MCTS flat-score, etc.). Residuals in §4.
- **Cognitive-layer redesign: APPLIED + deployed** (2026-06-28). A paired ablation showed the
  in-session cognitive stack ≈ a stripped baseline at ~1.8× latency; advisory/ungrounded layers
  are default-OFF via module constants (§3). Cross-session memory DOES earn its keep (Track B).
- **6-agent improvement review (2026-07-07): 24 of 27 items DONE**, then #6 (pin durable) and #7
  (accept lean state — no trim) CLOSED later the same day. Only **#5** (agent.py hot-path refactor,
  deferred to a focused session) and the **B3-gated #4/#27b** remain. Full unit suite **6587 passed /
  11 skipped / 0 failed** after the 2026-07-07 correctness/security sweep (+~70 tests).
- **Live validations (2026-07-07):** KV pin confirmed holding in prod (byte-identical
  stable-prefix hash across a request's turns); B3 idle-loop ablation ran a first pass AND a deeper
  3-arm run — idle loops are **proven productive** (self-play + reflection lessons; control 0), but the
  fact-recall probes hit a **ceiling** (97% both arms — memory saturates them), so the "does idle output
  improve *outcomes*" and "frontier vs uniform self-play" verdicts need a harder task battery (§4A #4/#27b).

Deployment: single long-running asyncio process on macOS (Darwin), served on :8000, upstream
local Qwen3.6-35B-A3B "heretic" on llama-server :8088. RAM-tight (36GB box, llama ~22.6GB
wired). Repo is versioned on another server (local git intentionally absent).

---

## 2. Operational reference (live agent)

**Process / flags.** `python -m src.ghost_agent.main --port 8000` under a **root launchd job**
`/Library/LaunchAgents/com.local.ghost-agent.plist` (**KeepAlive=true**). Live flags (2026-07-07):
`--verbose --deep-reason --smart-memory 0.9 --max-context 240000 --api-key "" --mandatory-tor
--autoadvance-idle --enable-metacog --metacog-mem-high 98 --metacog-mem-floor-mb 300
--visual-nodes http://127.0.0.1:8088|Eva --image-gen-nodes http://100.122.46.101:8000|Jetson`.
Env: `GHOST_HOME=/Users/vasilis/Data/AI/Data/`, `GHOST_CRITIC_ASYNC=1`, `GHOST_CRITIC_NO_THINK=0`,
`GHOST_PIN_TOOL_SCHEMAS=1`. cwd `/Users/vasilis/Data/AI/Agent`. Verifier is ENABLED; postmortem is
deliberately OFF. The launcher exec line (out-of-repo `bin/start-ghost-agent.sh`) is the only
flag truth — this list can drift; check `GET /api/health` `config` for the resolved reality.

**⚠️ Supervisor gotcha.** A plain `kill` of prod is undone within ~9s by launchd (parent pid 1).
To actually stop prod for an isolated run, the operator must disable the launchd service
(`sudo launchctl bootout …` / unload the plist). Re-enable it afterward to restore auto-restart.

**Restart WITHOUT the launchd service** (must use the **venv python** — the bare homebrew python
lacks `uvicorn`; Tor on :9050 must be up for the `--mandatory-tor` boot gate):
```bash
cd /Users/vasilis/Data/AI/Agent
export GHOST_HOME=/Users/vasilis/Data/AI/Data/ GHOST_CRITIC_ASYNC=1 GHOST_PIN_TOOL_SCHEMAS=1
/Users/vasilis/Data/AI/.agent.venv/bin/python -m src.ghost_agent.main --port 8000 \
  --upstream-url http://127.0.0.1:8088 --visual-nodes 'http://127.0.0.1:8088|Eva' \
  --image-gen-nodes 'http://100.122.46.101:8000|Jetson' --verbose --deep-reason \
  --smart-memory 0.9 --max-context 240000 --api-key '' --mandatory-tor --autoadvance-idle \
  --enable-metacog --metacog-mem-high 98 --metacog-mem-floor-mb 300 \
  >> /Users/vasilis/Data/AI/Logs/ghost-agent.log 2>&1 &
```
A manually-started prod is **unsupervised**; kill it before re-enabling the launchd service to
avoid a :8000 bind conflict.

**Drive a request** (model name validated — must be `qwen-3.6-35b-a3`):
```bash
curl -s -m 180 -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"model":"qwen-3.6-35b-a3","messages":[{"role":"user","content":"…"}],"stream":false}'
```
Reply at `choices[0].message.content`. **Introspect health:** `GET /api/health` (X-Ghost-Key)
returns rss/uptime/tasks/foreground counters/`biological_watchdog_alive`/`memory_system_loaded`/
scheduler jobs + the resolved config. Two silent-failure detectors: `memory_system_loaded=false`
= degraded boot (all biological phases dead); `biological_watchdog_alive=false` = daemon died.

**Logs.** Live pretty-stream (ANSI, appends across restarts):
`/Users/vasilis/Data/AI/Logs/ghost-agent.log` (errors: `…/ghost-agent.err`). App logger
(WARN/ERROR, non-pretty): `$GHOST_HOME/system/ghost-agent.log`. Boot dumps a "Resolved Config"
block + writes `$GHOST_HOME/system/last_config.json`. `kill -USR2 <pid>` dumps live asyncio
tasks + await stacks (hunts silently-parked coroutines).

**Test suite** (~6500 tests / ~3 min, must be green):
`GHOST_API_KEY=test-key PYTHONPATH=src /Users/vasilis/Data/AI/.agent.venv/bin/python -m pytest tests/ -q`.
`FORCE_COLOR` unset for two env-sensitive thinking-loop tests. Never wipe memory / `$GHOST_HOME`.
`run_selfhood_functional.sh` is DESTRUCTIVE (`rm -rf` live selfhood) — never on live GHOST_HOME.

**RAM reality.** 36GB box; llama-server ~22.6GB (wired/mlocked → protected from the OOM killer).
With prod up only ~150–660MB physical free + ~1GB swap. A throwaway agent boots to ~630MB RSS in
~12s; ONE throwaway + prod fits, but sustained multi-agent idle work is risky — **stop prod first**
for ablations. Swap stays ~950MB free once a throwaway is up; abort a run if swap_free < 250MB.
`malloc_trim` is Linux-only (no-op on Darwin). RSS watchdog (#3) is opt-in via `GHOST_MAX_RSS_MB`
(default off).

**Conventions.** Prefer closing loops over new modules. A "bug" needs a concrete failure scenario;
an "inefficiency" needs measured before/after. Any change adds tests in `tests/` + updates HTML
docs in `docs/`. Flag/env changes need a manual relaunch. Logging: `pretty_log` + distinct icons;
the operator watches the live stream; `logger.warning`/`error` auto-render.

---

## 3. Cognitive-layer redesign (2026-06)

> Cited by `core/agent.py:73,104` and `docs/algorithms/metacognition.html`. The original doc was
> lost (no local VCS) and reconstructed 2026-07-07 from the surviving toggle comments + the memory
> ledger. Percentages are historical; the **decisions + re-enable criteria** are load-bearing.

**The finding.** A paired, time-matched ablation (`scripts/ablation_paired.py`) killed the
shared-upstream contention confound and showed the full in-session cognitive stack did **not** beat
a stripped baseline: trivial suite both 100% (the earlier "45%" was a contention artifact); hard
suite **full 78% vs thin 80%, McNemar p=1.0, full ~1.8× latency** (agent RSS grew ~270MB→~2GB over
2.3h, OOM at repeat 5). A 6-agent review diagnosed every layer as one of: **(a) advisory not
load-bearing** (MCTS "strong hint, not a mandate"; recalled skills are prose never executed;
selfhood narrative; confidence logged-only; RRF computed a ranking then discarded it via fixed
per-source budgets); **(b) ungrounded signal** (MCTS scored self-prediction of un-executed actions;
the dual-solver arbiter sampled 2 completions, threw both away, dispatched the original — dominant
latency, 0 answer changes; the grounded `hypothesis.py` loop was dead code); **(c) open loop**
(postmortem defects read only by an operator tool; router+PRM shipped untrained→escalate-all;
nothing graduated). **Worst single bug:** the temperature policy sampled AWAY from correctness —
graded factual Q&A classified "conversational" → temp 1.0 + presence_penalty 1.5.

**Applied (toggles are module constants in `core/agent.py`):**

| Change | Site | Status |
|--------|------|--------|
| Greedy sampling for graded turns | `_is_factual_query` + `FACTUAL_SAMPLING_PARAMS` | live |
| RRF emits by fused score under ONE global budget (12k→4k) | `bus.py _format_markdown` | live |
| Relevance-gate every tier (dropped `"user"` graph seed; vector distance-gate; episodic threshold) | `bus.py` fetchers | live |
| Metacog dual-solver arbiter OFF | `_METACOG_ARBITER_ENABLED = False` | **OFF** |
| Grounded hypothesis test→evaluate→survive loop wired | `_HYPOTHESIS_GROUNDING_ENABLED = True` | live (needs `--deep-reason`) |
| MCTS turn-start hint OFF | `_MCTS_TURNSTART_ENABLED = False` | **OFF** |
| Normalized graduation + discriminative credit + mints a `proposed` macro | `skills_auto` | live |
| Router trains/loads at startup (not escalate-all) | `router/trainer.bootstrap_router` | live |
| Selfhood wake-up prefix OFF; workspace prefix gated on active project | `_SELFHOOD_PREFIX_ENABLED = False` | **OFF** |

**Re-enable criteria (why each OFF toggle is parked, not deleted):**
- `_MCTS_TURNSTART_ENABLED` — only with an **execution-grounded** value fn (not self-prediction).
  Its intended grounded replacement (verifier-judged best-of-N that SUBSTITUTES the winner) landed
  2026-07-07 as the async-critic bounded repair (§5 #18).
- `_SELFHOOD_PREFIX_ENABLED` — the prefix injects no facts/tools/constraints (pure token cost). The
  load-bearing selfhood path is the cross-session memory substrate (Track B), which IS proven.
- `_METACOG_ARBITER_ENABLED` — net-negative as built; superseded by #18.

**Do NOT re-enable an OFF layer** without meeting its criterion or a fresh paired-ablation win.
The default-OFF state is the measured-neutral configuration, not an accident.

**Track B / B3.** Cross-session MEMORY tiers are PROVEN (Track B: 98% recall treatment vs 0%
control). The pure-idle loops (dream/self-play, reflection critique, skills-auto graduation, PRM)
were unadjudicated until B3 — see §4/§6: B3's first live pass (2026-07-07) proved the self-play
loop productive; the deeper "does idle output improve outcomes" question is still open.

---

## 4. WHAT REMAINS TO DO

Everything below is open. Start here. Grouped: (A) improvement-review partials/blocked,
(B) static-hunt deferred findings, (C) functional-hunt deferred findings, (D) the B4
outcome-battery design (the harder task battery #4/#27b are blocked on — designed
2026-07-09, awaiting implementation).

### 4A. Improvement-review items not fully closed (from the 6-agent review)

- **#5 — agent.py guard-seam refactor (PARTIAL, high value; steps 1-2 of 4 DONE).** The seam is
  established (`core/stream_guards.py`); step 1 (tool-call parser → `_parse_assistant_tool_calls`)
  shipped + live-validated 2026-07-08; **step 2 (tool guard/dispatch/result pipeline →
  `_dispatch_and_process_tool_batch` + the `TurnState` dataclass) shipped 2026-07-09** — the
  ~1,300-line region was extracted VERBATIM against a TurnState designed from an AST capture
  analysis (16 MUTATED_FIELDS incl. cross-iteration latches like `_request_sys3_fired_once` that
  previously survived between turns in handle_chat's frame; finally-based repack keeps state exact
  on raising tool paths; the region was the turn-loop tail so its `continue`/`break` became a
  boolean return). Suite 6795 green; direct tests `tests/test_dispatch_pipeline_extraction.py`;
  3 stale source-inspection tests updated. **LIVE-VALIDATED 2026-07-09** (operator restarted prod):
  a real file-write+read request dispatched through the new method — and the NATIVE tool_call
  corruption repair fired mid-request and recovered (the hairiest branch, exercised live), verifier
  CONFIRMED 100%, exact bytes on disk. **Step 3 (finalization chain → `_finalize_and_return` +
  `FinalizeState`) shipped 2026-07-09 (later):** the ~950-line post-turn-loop tail (scrubbers →
  deferred Perfect-It → verifier gate/calibration → competence+skill credit → episode recording →
  correction stash → return) extracted VERBATIM with ZERO control-flow rewrites — the region ends
  in handle_chat's own `return`, nothing after it reads locals, so FinalizeState is read-only (20
  fields, no repack). One pre-bind added (`payload = None` before the turn loop) so FinalizeState
  construction can't hit an unbound name on the deterministic-dispatch exit path. Suite 6801 green;
  direct tests `tests/test_finalize_extraction.py` (6); 3 stale source-inspection tests updated.
  **Live-validated on a throwaway agent** (same code path; file-write probe → exact bytes on disk,
  verifier gate — which lives INSIDE the extracted chain — CONFIRMED 100%, zero errors); prod picks
  it up at its next restart. Remaining: (4) final-generation streamer closure (~502 lines, heaviest
  closure coupling — extend TurnState from a fresh capture analysis). Same protocol.
- **#6 — `GHOST_PIN_TOOL_SCHEMAS` durable — DONE 2026-07-07.** Durability was already in place: the
  launcher (`bin/start-ghost-agent.sh` line 231) exports `GHOST_PIN_TOOL_SCHEMAS=1`, which launchd
  runs — so prod is durable across restarts. Confirmed the pin is **holding live**: a per-turn
  `prefill cache · stable-prefix h=… len=…` log line, hash stable within a conversation. The **code**
  default stays OFF deliberately — flipping `os.getenv(..., "0")→"1"` reorders prompt assembly for
  every non-prod launch and trips 8 integration tests that pin the unpinned message layout (tried,
  reverted); durability belongs in the launcher, not the global code default. Measurement caveat:
  precise `n_past` quantification needs llama-server `--metrics` (currently off) — not enabled to
  avoid restarting the OOM-protected prod LLM; the `--swa-full` caveat in the launcher comment is
  moot here (Qwen3.6-35B-A3B is full-attention MoE, not SWA). Code comment updated to record the
  decision.
- **#7a/b — tool-schema diet — CLOSED 2026-07-07: accept the current lean state (operator decision).**
  Measured all 35 advertised tools live: the finding's "10.6KB/5.7KB/4KB" were **full** schemas
  (desc+params); the DESCRIPTIONS are already lean from the #7a/b/c work — `manage_projects` desc is
  3.3KB (not 10.6KB), `browser` 797 chars, `file_system` 621. Totals: desc 19.5KB, **params 31KB**
  (~13.5k tokens full). The remaining bulk is PARAMETER schemas, and it's necessary contract:
  `manage_projects` is one tool exposing **23 actions across 34 fields** (each field explains a
  non-obvious arg — `ledger`, `constraints`, `count`). Trimming risks the model not knowing an
  action/arg exists, for token savings the **KV pin already amortizes** across a request's turns. The
  ~11 "selfhood/meta" tools (introspect, self_state, dream_mode, self_play×3, skills×3) are
  functionally DISTINCT — folding behind one `self_report()` would conflate different operations and
  hurt selection. **Decision: no trim.** The descriptions are already lean; the residual size is real
  contract; a blind cut is net-negative risk (same shape as #6's declined code-default flip). If ever
  revisited, it must be a live A/B (trim → battery of tool-selection/arg-filling prompts → compare),
  never a headless cut — but the operator has accepted the lean state, so this is CLOSED.
- **#27b — CLOSED 2026-07-09 (23:30): default FLIPPED to uniform** (`--frontier-selfplay` now
  opt-in) after frontier tied uniform in both instrumented ablations (B3 2v2; B4 equal in all 4
  repeats). PRM STAYS — self-play productive in 3/4 repeats per arm; "delete PRM" never triggered.
  Re-enable criterion: a run where frontier out-yields uniform. Original findings below for
  history.
- **#27b — PRM keep/delete verdict (frontier sub-arm ran, INCONCLUSIVE 2026-07-07).**
  The frontier-vs-uniform self-play sub-arm executed (deeper B3, 3 arms × 2 repeats). Result: on the
  metric that matters — *self-play* lesson yield — frontier and uniform **TIED (2 vs 2)**. Frontier's
  4-vs-2 total edge was entirely 2 *reflection* lessons, which are orthogonal to `--frontier-selfplay`.
  So at N=2 repeats the frontier selection is a **wash** — no evidence it out-yields uniform seeding.
  "Delete PRM" is still **not** triggered (self-play loop is productive either way), but "keep frontier
  BECAUSE it beats uniform" is **unproven**. Verdict needs more repeats + a HARDER task battery (see #4).
  **→ Battery designed 2026-07-09 with a pre-registered frontier verdict: §4D.**
- **#4 — UPDATE 2026-07-09 (B4 full run executed, §6): outcome question STILL open but now with a
  diagnosed dependency chain — (a) battery difficulty unsolved for this model below expert tier
  (re-ceilinged at 97% under run conditions; pilot difficulty was partly cross-pass memory
  interference — pilots must boot fresh per pass); (b) mediation ≈ 0: playbook lessons never
  surface in task-shaped probe turns → fix lesson RETRIEVAL ROUTING before any bigger run;
  (c) dream definitively needs a trajectory-shaped seed source (code change, not protocol).
  The next step on #4 is those two code changes + an expert-tier battery, NOT more repeats.
  Original 2026-07-07 finding below for history.**
- **#4 — B3 deeper run — EXECUTED 2026-07-07 (methodological result, not a clean win).** Ran 3 arms
  (treatment/frontier, treatment_uniform, control) × 2 repeats on 18 enriched seeds; added McNemar +
  frontier-yield to the harness. Findings: (1) **idle loops confirmed productive** — treatment 4
  lessons (2 self_play + 2 reflection), uniform 2, control 0; reflection fired this time (first pass
  only got self-play). (2) **Probe-outcome McNemar is a CEILING ARTIFACT** — fact-recall probes sit at
  ~97% in BOTH arms because memory is ON in both, so they cannot detect idle-loop value (p=1.0 is
  uninformative, not negative). The "does idle output improve outcomes" question needs a **harder task
  battery** where recall isn't the bottleneck — the current Track-B recall probes are the wrong
  instrument. (3) **Dream STILL didn't fire** even with 18 richer seeds — fact-shaped seeds aren't
  enough; it likely needs failed/diverse *trajectories*, not stored facts. Report:
  `ablation_out/trackb3-20260707-191216/`. Remaining: harder-task probe suite + more repeats for a
  real outcome-improvement + frontier verdict. **→ The harder battery is now DESIGNED (2026-07-09):
  see §4D for the full protocol; what remains is implementing `trackb4_tasks.py` +
  `ablation_trackb4.py` and running the pilot + overnight run.**
- **#1 — version control:** SKIPPED per operator (repo versioned on another server). No action.

Also deferred within done items: #3 launchd supervisor plist in-repo (out-of-repo launcher);
#9 warm-runner browser tier (long-lived in-sandbox Playwright process); #27c persistent token→node
graph inverted index (the forgetting pass + node-cache landed).

### 4B. Static bug-hunt deferred findings (source review — still open unless marked RESOLVED)

Severity in parens. Many are latent (no prod caller), multi-process (single-tenant today), or
model-behavior edges.

- **`current_project_id` cross-conversation race (projects + api)** — **RESOLVED 2026-07-07.** #22
  (turn serialization) closes the chat-turn-vs-chat-turn window; the residual was the stateless
  `/api/upload` + `/api/download` endpoints, which carry no conversation context and read the racy
  global (a concurrent switch/reconcile could land an upload in another conversation's sandbox).
  Fix: `project_scoped_sandbox(..., explicit_project_id=)` + a `?project_id=<id>` query param on
  both endpoints → a client scopes race-free; the global stays the fallback when absent. Tests:
  test_upload_project_scope.py (6). Docs: file_system.html, api/routes.html. (Full per-conversation
  threading of every `record_*` call remains the deeper option, but the exploitable API surface is
  now closed.)
- **workspace `current_project_id` event-stamping race** — **RESOLVED 2026-07-09 (and #22 was NOT
  sufficient).** Root cause found on the confirm pass: there are TWO project-id fields —
  `context.current_project_id` (sandbox scoping; what #22's serialization and the 2026-07-08
  `pinned_project_context` protect) and `workspace_model.current_project_id` (what every `record_*`
  actually stamps from). So (a) idle autoadvance mis-stamped its command outcomes with the LAST chat
  turn's project **deterministically** (the pin covered the wrong field), plus a live race when a
  user turn overlapped an in-flight tick; (b) dream self-play's temp agent (its OWN semaphore, so
  unserialized) set the shared field to `""` mid-flight and recorded synthetic outcomes into the
  real activity log; (c) `manage_projects autoadvance` with an explicit foreign project_id stamped
  the chat's project. Fix: task-local ContextVar override in `workspace/model.py` read by every
  stamp site (`set_event_project` bound by handle_chat at the stamp-sync site; `pinned_event_project(pid)`
  wrapping the idle tick and the explicit-project batch); dream detaches the shared model entirely
  (`isolated_context.workspace_model = None` — all record/prefix sites guard on None). Real chat
  turns and the scheduler outcome write were already safe (serialization / no-await-gap). Tests:
  test_workspace_event_stamping.py (8, incl. the reproduced interleave). Docs:
  core/workspace_model.html, core/project_advancer.html, core/dream.html.
- **memory projects metadata split-lock + skills cross-process lock** — **RESOLVED 2026-07-07.**
  `projects.py` is SQLite-backed, so the RMW (`append_ledger`/`set_ledger`/`set_config_value`) now
  routes through `_atomic_metadata_update`, which runs SELECT→mutate→UPDATE inside a single
  `BEGIN IMMEDIATE` transaction (grabs the write lock before the read → cross-connection
  serialization; a competing writer waits on `busy_timeout` instead of clobbering). `skills.py` (JSON)
  now writes a **PID-unique** temp and wraps write+`os.replace` in an `fcntl` advisory lock on a
  sibling `.lock` file (mirrors `frontier.py`, graceful no-op without fcntl). Tests:
  test_memory_crossproc_locking.py (7). Docs: memory/projects.html, memory/skills.html.
- **graph.execute_graph_compression resurrects expired facts** — **RESOLVED 2026-07-07.** The
  node-merge now snapshots every triple touching `old_node` on either side, rewrites *both* endpoints
  (so an `old→old` self-loop migrates to `new→new` instead of being swept by the `object = old_node`
  delete), and carries `valid_from`/`valid_until` through a new `_merge_triplet_row` helper — a
  superseded fact stays expired instead of re-entering as current, weights sum without double-counting,
  and a temporal merge is current-wins (either side current ⇒ current; both expired ⇒ later expiry +
  earliest `valid_from`). Still unwired (only a no-op stub in `dream.py` calls it) — hardened before
  wiring. Tests: test_graph_compression_temporal.py (7). Docs: memory/graph.html.
- **vector smart_update template over-match + correct_fragment id-collision** — **RESOLVED
  2026-07-07.** smart_update still computes `dist<0.50` but now ALSO requires the neighbor's
  extracted subject key to agree before deleting (`_subject_key`: "User's favorite color is blue" →
  `favorite color`), so a distinct template-sibling ("favorite food") is kept while a genuine
  restatement still collapses; template-less paraphrases fall back to distance-only (preserves dedup).
  correct_fragment switched `add()` → `upsert()` so a replacement whose md5-of-text collides with an
  existing id always lands. Tests: test_vector_memory_dataloss.py (8, reproduction confirmed on
  revert). Docs: memory/vector.html.
- **prm binary-floor gates continuous training + train↔serve feature skew** — **RESOLVED (gate) +
  SURFACED (skew) 2026-07-07.** The training-viability floor is now mode-aware: continuous mode
  (the default) requires both regimes represented (≥1 success-side + ≥1 failure-side sample, so
  all-PASSED/all-FAILED still bail) plus a `min_label_std` variance floor, and does NOT re-impose the
  binary `min_class_fraction`; binary mode keeps the fraction floor. A mostly-failing corpus with a
  few high-value anchors (~3% binary-positive) now trains instead of false-bailing "class imbalance."
  The feature skew is now SURFACED: `TrainerReport.feature_skew_warning` flags any
  `SERVE_TURN_START_INERT_FEATURES` (steps_so_far, failures_so_far, tool_used/failed_this_turn) that
  carry training variance, so train accuracy isn't read as deployed discrimination. Full skew fix
  (score at turn start / drop those columns) is still the training-signal redesign — left open,
  relevant to #27b. Tests: test_prm_binary_floor_and_skew.py (6). Docs: algorithms/prm.md.
- **reflection selection oldest-first** (low) — `Reflector.run` is oldest-first within a tick (a
  recency window reaches fresh failures faster); `_truncate` head-keeps `tc.error`/`failure_reason`
  (drops the tail exception of a traceback); a diagnosis with "plan:" is truncated at that word.
  (The non-persistent-dedup half was RESOLVED 2026-07-04.)
- **distill outcome-heuristic false positives** (low-med) — the tool-error heuristic's bare
  substrings ("exception"/"traceback") match benign read content; `[ATTEMPT_ABORTED_*]` regex is
  searched in the user-facing final_response. Bounded today by the 3-repeat/two-signal gates.
  (Structured `ToolCall.error` on the chat path landed 2026-07-07 as #27d — improves this.)
- **agent.py correction-lookup fingerprint mismatch on prepended turns** — **RESOLVED 2026-07-07.**
  `_response_fingerprint` now peels leading banner blocks (`_strip_leading_banners`) before hashing —
  the three deterministic prepends (async-verdict correction ⚠️, clarifying-question lead-in,
  autonomous-progress digest) all share a `\n\n---\n\n` separator and stack in front of the body, so
  the banner-less core is invariant and stash- (body) vs lookup-time (banner+body) hash the same key.
  The bound keeps a genuine long intro before a markdown rule intact; peeling loops so stacked banners
  all strip. Closes the silent drop of the "confidently wrong" calibration-negative + FAILED promotion
  on hedged/corrected turns. Tests: test_correction_fingerprint_banners.py (11, incl. e2e promotion
  gate). Docs: self_improvement.md.
- **streamed-turn calibration gap** (low) — streamed final generations bypass the finalize tail →
  write no calibration JSONL pair, log a competence-only `below=`; verdict deferred to a next-turn
  banner. Async-verification design gap, not a finalize skip.
- **agent.py trajectory tool-result pairing on id-less duplicates** (low-med/contingent) — two
  same-named tool calls in one turn with empty ids collide on the `name` key in `pending_calls` →
  one result dropped (blank ToolCall on disk), Signal-3 error-repeat promotion undercounted.
  Contingent on local models streaming id-less tool calls. Needs an index-fallback key.
- **acquired_skills / composed / qwen_bridge** (low/med) — AcquiredSkillManager re-instantiated
  per call, each its own RLock (lost failure_count increment under concurrency); composed macro can't
  thread step N output into step N+1 ($var only resolves against initial params — feature gap);
  qwen_bridge `_run_coro_blocking` runs each native coroutine on a fresh loop (cross-loop error if a
  native tool caches a loop-bound client — agent_qwen.py variant only).
- **file_system replace bad-byte write-back corruption** — **RESOLVED 2026-07-07.** The replace
  read + guarded write + streaming path now use `errors="surrogateescape"`, so untouched non-UTF-8
  bytes round-trip to their exact originals instead of being persisted as U+FFFD; the
  syntax-regression guard fails open when `ast.parse` raises `UnicodeEncodeError` on a lone
  surrogate. Tests: test_replace_bad_byte_roundtrip.py (5). Docs: tools/file_system.html.
- **file_system + darkweb SSRF-on-redirect / body-cap** — **RESOLVED 2026-07-07.**
  `tool_download_file` disables auto-redirect and follows hops manually, re-validating each
  `Location` with the SSRF guard (`_download_redirect_target`, bounded at `_MAX_DOWNLOAD_REDIRECTS`).
  The onion fetch (`_fetch_raw_html`) now STREAMS the body (`iter_content`/`iter_bytes`) and stops at
  `_MAX_ONION_BODY_BYTES` instead of materializing `r.text` whole. Tests: test_download_redirect_ssrf.py
  (9). Docs: file_system.html, darkweb_search.html. Browser SSRF CORE was RESOLVED 2026-07-04
  (in-sandbox route interceptor); the two residuals are now **RESOLVED 2026-07-07** — the in-sandbox
  interceptor (`_install_ssrf_guard` in browser.py's embedded runner) now (a) blocks `file://` unless
  the `os.path.realpath` stays within the `/workspace` sandbox root (component-wise `commonpath`,
  fail-closed on unresolvable), and (b) in non-Tor mode re-resolves each request host via
  `getaddrinfo` and aborts on any internal IP (defeats DNS-rebind of a subresource); over Tor the
  lookup is skipped (no leak, can't route internal anyway). Tests: test_browser_ssrf_residual.py (24).
  Docs: tools/browser.html.
- **execute nits** (low) — `_inline_py` `-c` body detector false-block on a chained command reusing
  the delimiter quote (annoyance, never wrong execution); `stateful=True, args=[…]` drops argv[2:];
  file-not-found retry re-runs side-effecting commands (substring heuristic); single global stateful
  kernel (documented tradeoff).
- **tool_failure loose FATAL patterns** (low) — `invalid.?(arg|param|schema)` / `tool.*not found`
  checked before DIAGNOSTIC, so `ValueError: invalid argument…` / `FileNotFoundError: 'tool.py' not
  found` get marked FATAL instead of self-correctable. Also: dead retry helpers
  (`get_retry_delay`/`should_retry`/`MAX_RETRIES`, off-by-one) — wire-or-delete.
- **sandbox/docker.py** (med/rare + high-on-Linux) — remove-while-exec race (in-flight command dies,
  self-heals; do NOT auto-retry — commands aren't idempotent); no client-side deadline on probes/exec
  (a wedged daemon hangs a worker); **Linux exec-user vs root-provisioned env** (Playwright/Chromium
  in /root mode 700, sudo refuses unknown uids — masked on macOS which execs as root; forces a full
  re-provision to fix — do when a Linux deployment matters).
- **router serve-only / scaling features** (low) — context_turn_coupling computed at serve but never
  trained (column all-zeros, inert); code_fence_count/coding_language raw not log1p; multi_step
  uses unanchored substrings. Needs a schema bump + retrain.
- **optim GEPA pipeline** (med, offline/opt-in) — not dspy-compatible: `tuner.compile` handed plain
  `TrainExample`s never converted to `dspy.Example` (real dspy run crashes, masked by the test mock);
  signature input-field name mismatch; the A/B `_ab_runner` feeds the instruction as a bare system
  prompt (not how the agent embeds it at inference → OOD gate). Needs a dspy env. (Adoption-ordering
  half RESOLVED 2026-07-04.)
- **scripts measurement robustness** (low) — `ablation_eval` globs a fixed report dir and folds
  stale/foreign-model JSONs into the verdict + treats (task,repeat) as independent in Wilson CIs
  (over-narrow); `selfhood_functional_test` Section C/G non-atomic RMW on the live store + Section D
  hardcodes :8088; `load_tokens` O(n²) filler; timestamp collisions. (**Note:** `ablation_trackb3.py`
  had a report-builder bug — FIXED 2026-07-07.)
- **interface external GPU servers** (med, home-lab threat model) — TTS/STT + image-gen HTTP servers
  bind 0.0.0.0 with NO auth on expensive endpoints (LAN resource exhaustion) + unbounded reads +
  blocking model calls on the async loop; needs a shared-key check coordinated with the clients. The
  clockwork desktop client parses SSE with `aiter_text` on non-line-aligned chunks (token misframing),
  ships a placeholder key, leaks camera/QTimer on close.
- **entrypoint / misc lifecycle nits** (low) — `--no-memory` leaks a `/tmp/ghost_no_memory_*` dir per
  boot; `_host_signal_to_bus` logs hardcoded 85/90 thresholds not the configured values; unbounded
  numeric CLI args; scratchpad connections not closed; episode float-epoch timestamp (degraded
  ranking); `--prm-online-update` holdout slice contains the just-trained sample (leak);
  `_reflected_trajectory_ids` unbounded (RESOLVED 2026-07-04 — now persisted+bounded);
  utils/telemetry `stop()` can swallow caller cancellation.

### 4C. Functional bug-hunt deferred findings (live behavior — still open)

- **[affordance] `workspace` tool has no `search` action** — **RESOLVED 2026-07-09.** The guess is
  now a real action: `search` (alias `recall`, unadvertised to keep the schema lean per #7) does
  IDF-weighted keyword search over the whole activity log (`WorkspaceActivity.search`, mirroring
  selfhood's `search_my_past`; matches summaries, kinds, project ids AND payload values, so
  filenames/URL components hit). Schema advertises `search` + a `query` param; near-miss arg names
  (`q`/`text`/`keywords`) are absorbed instead of striking; the no-match reply redirects to the
  `recall` tool / `manage_projects` (which also nudges the recall-routing item below). A consistency
  test pins the schema enum ⊆ `_VALID_ACTIONS` (they are two sources of truth). Tests:
  test_workspace_search.py (14). Docs: tools/workspace.html.
- **[behavior] "project X" recall-routing variance** (low, model-dependent) — "when does project
  Kestrel ship?" made the model check the *projects* tool and give up, while "search your memory for
  Kestrel" recalled it (0.76). Storage correct; nudge via tool description or a recall-fallback on
  empty project lookups.
- **[search] Yandex fails over Tor** (low, known) — per-exit-node reachability; circuit rotation per
  attempt already implemented; search succeeds via other backends. MEASURE across exit nodes before
  changing the backend set (see `tor-search-reachability` memory). Possible cheap win: shorter
  per-backend Tor timeout for known-flaky backends (needs measurement).
- **[response] long skill-list truncated** (low) — "list all skills" tabulated 24 built-in tools and
  ran out of budget before acquired/composed; verifier LATE-REFUTED. Check whether truncated-answer
  auto-continuation fires here, or nudge to summarize not tabulate.
- **[infra] smart-memory upstream 503 not retried** — **RESOLVED 2026-07-09.** Checked first: the
  HTTP layer DOES retry (worker-node failover, then one 2s retry on any 5xx in `_do_chat_completion`)
  — but a main-node TIMEOUT was never retried (falls to the generic handler), and on final failure
  the consolidation was lost **permanently and invisibly**: the journal item was already `pop_all`'d,
  the task swallowed the exception with a bare `logger.error`, nothing re-queued, and the
  adaptive-threshold observation was skipped too. Fix at the task level: `run_smart_memory_task` now
  raises `RetryableConsolidationError` on upstream-transient failures (5xx/timeout/connection,
  classified by `memory.journal.is_upstream_transient`) BEFORE any memory write; `process_journal_queue`
  re-queues the item with a bounded `retries` counter (`JOURNAL_MAX_RETRIES=2`), with visible WARNING
  lines on both re-queue (🔄) and final drop (🔶). Definitive failures (4xx/parse) keep log-and-drop —
  a re-run would fail identically. Post-mortem items share the drain loop and could adopt the same
  classification later (not done — scoped to the reported item). Tests: test_smart_memory_requeue.py
  (8). Docs: memory/journal.html.
- **[infra] torch leaked-semaphore at shutdown** (low, cosmetic) — `resource_tracker: 1 leaked
  semaphore` on every restart (the local embedder). Check the embedder subprocess/pool teardown.
- **[coding] huge-reasoning no-file-spec** (reviewed, no fix) — model emits only prose reasoning, no
  spec; salvage logic already present. Genuine model-behavior edge.

### 4D. B4 outcome-battery design (2026-07-09) — the harder task battery for #4/#27b

> **IMPLEMENTED 2026-07-09 (same day):** `scripts/trackb4_tasks.py` (22 probe candidates across all
> 7 seeded clusters + the held-out web_automation far ring; 8 seeding tasks, 4 easy / 4 hard on the
> weak clusters; every task gated self-consistent) + `scripts/ablation_trackb4.py` (seeding phase,
> mediation capture, task-stratified sign-flip test, log-based dream-gate instrumentation,
> `--pilot` calibration mode emitting `b4_battery.json`; arms carry `--smart-memory 0.9`). Headless
> tests: `tests/test_trackb4_battery.py` (74). Docs: `scripts/ABLATION.md` §Track B4. What remains
> is OPERATOR EXECUTION: the ~2 h pilot, then the ~11 h overnight run (prod stopped, §2 gotchas).
> Note the pilot band is implemented as "neither all-pass nor all-fail over `--pilot-repeats` (3)"
> — the honest binary-sample version of the [0.3, 0.7] target below; extend the candidate pool if
> fewer than ~18 survive.

**Problem.** B3's probes are fact-recall string-matches; memory is ON in every arm, so both sit at
97% and McNemar cannot see idle-loop value (§4A #4: ceiling artifact). The open questions — "does
idle output improve OUTCOMES?" and "does frontier seeding beat uniform?" — need probes whose success
depends on *competence the idle loops can change*, calibrated off the ceiling.

**1. Task design.** ~36 candidate tasks across the self-play cluster families
(`data_analysis, regex_parse, sql, algo, bash, python_general, concurrency` — the
`classify_cluster` taxonomy) plus ONE held-out family (`web_automation`) that is never seeded, in
three transfer rings: **near** (isomorphic to a `challenge_templates` shape, fresh surface data),
**mid** (same cluster, new shape), **far** (held-out family). Every task is execution-grounded in
the `eval/behavioral.py` style: the prompt drives the LIVE agent to compute something over fixture
files in its sandbox and write a result artifact; a Python verifier checks the artifact
(`BehavioralTask.verify` — no LLM judge, no prose string-match). Generate instances from the
template bank at intermediate/advanced tier WITH twists (`na_rows`, `malformed_lines`,
`negative_values` …) — the exact difficulty axes self-play trains on. Contamination guard: probe
fixtures use a fixed harness seed; hash-compare each arm's generated challenge setups vs probe
fixtures and log any collision.

**2. Calibration pilot (the ceiling fix).** Run the candidate pool once against a CONTROL-configured
agent; keep only tasks with baseline pass-rate in **[0.3, 0.7]** (target ≥18 survivors, ideally
20–24); discard ceiling/floor tasks. This is the direct fix for p=1.0-by-saturation. Cost: one boot
+ ~36 probes ≈ 1.5–2 h.

**3. Arm protocol per repeat** (same 3 arms as B3; control keeps memory ON, time-scale 1):
- **Phase S — seeding (identical in ALL arms):** run ~8 seeding tasks picked to yield a mix of
  passes and REAL failures (3–4 known-hard). Purpose: (a) failed trajectories → reflection material;
  (b) auto-type memories → dream's entropy gate — **the arms must add `--smart-memory 0.9`**: B3's
  arms never passed the flag, and `run_smart_memory_task` is the ONLY writer of `type:"auto"`
  fragments, so dream's ≥3-auto-memories gate was unsatisfiable by construction (this, not seed
  richness, is the first thing to fix); (c) frontier clusters with VARIANCE — without weak-vs-strong
  cluster signal, `pick_frontier_seed` has nothing to exploit and frontier-vs-uniform is a coin flip
  by construction. Instrument at end of S per arm: auto-fragment count, failed-trajectory count,
  per-cluster stats.
- **Phase I — idle window:** 8 epochs × 70 s at time-scale 60. Snapshot `skills_playbook.json`
  before/after; count lessons by source.
- **Phase P — probes:** the calibrated battery once per repeat via `agent_behavioral_runner`;
  record passed, duration, steps, tool_calls, tool_errors (`trajectory_metrics`) + mediation
  evidence (below).

**4. Mediation instrumentation (the null-result differentiator).** An outcomes-null is
uninterpretable unless we know whether lessons ever ENTERED the probe turns. Per probe: diff
per-lesson `retrievals`/`last_retrieved_at` in `skills_playbook.json` around the request (the bus's
retrieval credit) + count `Memory Bus … Hydrated context` lines in the arm log. Note the routing
bias: the skill tier gets weight 2.0 only for procedural-intent queries — probe prompts must be
phrased as DOING tasks (they are), else lessons are down-weighted 0.5 by design. Report
`mediation_rate` = fraction of probe turns where ≥1 lesson surfaced. Pre-registered reading:
outcomes-null + mediation≈0 → fix retrieval routing (don't re-run bigger); outcomes-null +
mediation healthy → idle output genuinely doesn't transfer at this scale.

**5. Stats.** Primary: treatment-vs-control paired outcomes per (task, repeat) — report exact
McNemar as before BUT alongside a **task-stratified permutation test** (repeats within a task are
correlated; the (task,repeat)-independence flaw is the same one §4B flags for the Wilson CIs).
Power sketch: 20 tasks × 3 repeats = 60 pairs; at ~50% baseline a real +15–20 pp effect yields
~12–18 discordant pairs — enough for a directional verdict; 2 repeats is NOT (B3 had 1/1
discordant). Secondary: tool_errors/steps deltas (self-play lessons are often tool-idiom-shaped),
lesson yield by source, per-cluster and per-transfer-ring outcomes.

**6. Frontier verdict (#27b) — pre-registered.** Frontier's mechanism is picking WEAK clusters, so
the verdict metric is (a) self-play lesson yield and (b) probe outcomes ON the seeded-weak clusters,
frontier vs uniform. **KEEP frontier iff self-play yield ≥ uniform in ≥2/3 repeats AND weak-cluster
probe delta ≥ 0; otherwise flip the default to uniform** (simpler, one less moving part). PRM stays
either way — self-play productivity is already proven and the "delete PRM" trigger stays untripped.
If Phase S can't produce cluster variance, frontier has no signal by construction → uniform by
parsimony.

**7. Dream sub-experiment (piggybacked, not a gate).** Log the entropy-gate state each idle epoch
(auto-fragment count + dream's skip reason). Outcomes: gate satisfied + fired → dream lessons join
the yield counts; gate satisfied + never fired → NEW BUG, file it; gate organically unsatisfiable
even with `--smart-memory` on → journal decision on widening dream's seed source to trajectories
(the 2026-07-07 hypothesis).

**8. Budget / runtime.** Per arm-repeat: boot ~2 min + S ~16 min + I ~9 min + P ~20×2.5 min ≈
75–80 min → 3 arms × 3 repeats ≈ **11–12 h: an overnight run with prod stopped** (launchd bootout →
venv-python restore; §2 gotchas apply — sequential single throwaway agent, shared llama, probe
timeout 300 s). Reduced option (~8 h): run `treatment_uniform` only in repeat 1 + decide frontier
from yield-only mini-runs.

**9. Deliverables (next focused session).** `scripts/trackb4_tasks.py` (fixture generator reusing
`challenge_templates` + grounded verifiers in the `eval/behavioral.py` shape);
`scripts/ablation_trackb4.py` (extends the trackb3 driver: seeding phase, mediation capture,
stratified stats, `--pilot` mode); arm flags gain `--smart-memory 0.9`. Boot/teardown reuse
`ablation_eval` as today (trackb3 already imports it).

---

## 5. Completed ledgers

### 5A. 6-agent improvement review — 27-item board (2026-07-07)

DONE (24): #2 (600s self-stall), #3 (RSS watchdog+health), #8 (docker probe TTL), #9 (browser text
preview), #10 (one truncation policy + spill), #11 (line-ranged read), #12 (de-quadratic streaming),
#13 (episodic semantic recall), #14 (per-item RRF), #15 (vector tier scoring), #16 (retrieval-credit
inflation), #17 (bound autobiographical), #18 (async-critic bounded repair), #19 (reflection→bg
priority), #20 (`spawn_bg`), #21 (/api/health + config dump), #22 (serialize turns), #23 (workspace
save off-loop), #24 (dream heuristic keying), #25 (redesign doc + docs truth), #26 (test builders),
#27a (wire context_manager), #27c (graph forgetting + node-cache), #27d (ToolCall.error on chat path).
CLOSED post-review 2026-07-07: #6 (pin durable via launcher), #7 (accept lean — no trim). STILL OPEN:
#5 (agent.py refactor — deferred to a focused session), #4 + #27b (B3-gated; deeper run executed
2026-07-07, see §4A/§6). SKIPPED: #1 (git).
Full detail of each in §6 and in git history on the other server.

### 5B. Static bug hunt — 28 units, all CLEAR (2026-07-03/04)

utils · sandbox · workspace · tools-infra · tools-fs-exec · tools-web · tools-knowledge ·
tools-skills · tools-projects · memory · router · api · core-llm · core-planning · core-verify ·
core-projects · core-dream · core-agent(×2) · prm · reflection · selfhood · distill · eval · optim ·
skills_auto · entrypoint · interface · scripts. Each: every file reviewed for concrete-failure bugs;
confirmed bugs fixed with regression tests + HTML docs; full suite green on the session date.
Residuals in §4B. (Regression tests: `tests/test_bughunt_unit*.py`.)

### 5C. Functional bug hunt — 14 subsystems, all CLEAR (2026-07-04)

Core chat turn · Introspection & selfhood · Memory & profile · Code execution/sandbox ·
Filesystem/workspace · Web search & research · Tasks & scheduling · Skills · Projects & autonomy ·
Metacognition & verification · Deep-reason/planning · Vision & image-gen · DBA/database ·
API/interface. Real fixes shipped in units 1,3,5,7,11 (+ the post-sweep chess post-mortem and the
skills_auto graduation wiring). Residuals in §4C.

---

## 6. Session history (newest first)

### 2026-07-10 — public benchmark: GAIA harness hardened + readiness pilot PASSED (full run gated on HF token)
- **Goal:** post a real, representative public number (the "convert quality into credibility" move
  §1/§4 keeps circling). **Chose GAIA over SWE-bench**: GAIA (web-research + tools + files + short
  exact-match answers) matches this agent's built surface; SWE-bench is pure code-patching (the
  documented weakness) and its per-repo `pip install` fights the mandatory-Tor egress guard on a
  36GB box. GAIA has a public-answer validation split (165 q, 3 levels) to measure honestly before
  the held-out leaderboard.
- **Found existing May scaffolding** (`scripts/gaia_scorer.py` + `gaia_eval.py`) — hardened rather
  than rebuilt. (1) Consolidated the exactness-critical logic (official `question_scorer` + canonical
  prompt + `FINAL ANSWER:` extraction) into the dep-free scorer as the single source of truth; +23
  tests (`tests/test_gaia_scorer.py`) pinning number/list/string normalization, units/commas,
  last-marker-wins, and the empty-answer guard against the test-split "?" placeholder. (2) Added
  `--boot` (isolated throwaway agent, fresh GHOST_HOME, torn down after — prod untouched) and
  `--tasks-file` (offline pilot, no gated dataset) to `gaia_eval.py`; `--no-memory` default
  (defensible: GAIA tasks are independent, so cross-session memory can only leak across tasks).
- **Readiness pilot (8 GAIA-shaped known-answer q, isolated on :8046, prod up):** pilot #1 scored
  **0/8 despite 8/8 substantively-correct replies** — caught a real harness bug: the GAIA protocol
  was sent as a SYSTEM message, which the agent merges into its own large composed system prompt
  where the FINAL-ANSWER mandate loses salience → the model answered correctly in prose, never in
  template, extraction returned empty. **Fix: carry the protocol in the USER message** (standard for
  agents that own their system prompt). Pilot #2: **8/8 clean** (incl. list case-normalization and
  the multi-hop Booker question researched over Tor in 28.9s → 1918). Pipeline proven end-to-end:
  isolated boot → drive → Tor web research → FINAL-ANSWER extraction → official scoring → per-level
  report → teardown. Pilot measures pipeline health, NOT GAIA score (hand-picked stable facts).
- **BLOCKED on operator:** the real `gaia-benchmark/GAIA` validation set is gated —
  `huggingface-cli login` with a token that has accepted the agreement unblocks the 165-q run
  (`python scripts/gaia_eval.py --split validation --boot`). Then report by level; if strong, prep
  the held-out test-split leaderboard submission. Suite unaffected (+23 GAIA tests, all green).

### 2026-07-09 (23:30) — the three B4 verdicts ACTIONED (retrieval gate, frontier flip, dream seeds)
- **(1) Lesson retrieval domain-rescue (the write-only-learning fix).** Forensic replay against the
  B4 arm's REAL store: the arm's self-play lesson sat at embedding distance **1.056** from a
  matching task probe vs the strict `DEFAULT_RETRIEVAL_DISTANCE = 0.45` floor (~cosine 0.78 on
  normalized MiniLM) — a generalized lesson can never clear it against a concrete prompt, so the
  skill tier filtered it on all 96 probe turns. Fix in `skills._playbook_items_and_branch`: a
  candidate past the strict floor is admitted up to `_DOMAIN_RELAXED_DISTANCE = 1.25` when its
  `domains` metadata contains the query's cluster per `_explicit_query_cluster()` — which requires
  an explicit `CLUSTER_KEYWORDS` hit (NOT `classify_cluster`, whose python_general fallback would
  let small talk "match"); untagged lessons (reflection) derive a domain from their trigger.
  Semantic near-match OR domain match — never a blind dump. **Verified by replaying the real B4
  store through the fixed path: the lesson now surfaces for a python-shaped probe; small talk
  stays empty.** Tests: test_lesson_retrieval_domain_gate.py (8, incl. the forensic repro).
  Docs: memory/skills.html.
- **(2) `--frontier-selfplay` default flipped to OFF (#27b CLOSED).** Tied uniform on self-play
  yield in both instrumented ablations (B3 2v2; B4 equal in all 4 repeats) — parsimony wins; the
  machinery stays opt-in (re-enable criterion: a run where it out-yields uniform). getattr default
  in dream.py matched. PRM STAYS (self-play productive 3/4 repeats per arm). 100 frontier tests
  green unchanged. Docs: cli_reference.html, algorithms/dream_cycle.html.
- **(3) Dream trajectory seeding.** `trajectory_dream_fragments(context)` digests the newest
  trajectories (task/outcome/tools/first-error, `traj:` ids); `dream()` falls back to them when
  the auto pool is <3 and the watchdog eligibility gate mirrors the fallback. Merge/delete
  consolidation is DISABLED in trajectory mode (`traj:` ids must never reach collection.delete);
  the value is the heuristics harvest (`source="dream"`). Idempotency guard unchanged. Tests:
  test_dream_trajectory_seeds.py (8). Docs: core/dream.html.
- Deploy note: prod needs a restart to pick these up.

### 2026-07-09 (23:00) — B4 FULL RUN EXECUTED (3 arms × 4 repeats): triple-null, but every null is now DIAGNOSED
- Ran 18:08-22:52, clean (no holds, no swap pressure, 12 arm-boots). Report:
  `ablation_out/b4-20260709/`. Headline: outcomes p=1.0 — but unlike B3's ceiling artifact, the
  instrumentation localizes each failure mode precisely:
- **(1) The battery re-ceilinged at run time (97% all arms) despite pilot calibration at ~67%.**
  Diagnosis: the pilot reuses ONE agent across its 3 passes (cheap), so pass-2/3 failures were
  partly CROSS-PASS MEMORY INTERFERENCE (recalling pass-1's answer against pass-2's reseeded data);
  the run boots FRESH arms per repeat → cold agents solve the tasks ~always. **Calibration-protocol
  lesson: a pilot must match run conditions — boot fresh per pass.** Battery difficulty for this
  model remains unsolved below expert tier (37 candidates, two pilots, 8 survivors, all ceiling
  under run conditions).
- **(2) Mediation ≈ 0** (1 of 96 probe turns surfaced any lesson, despite 16/16 bus hydrations per
  arm): the ~1 self-play lesson per arm-run never entered a probe prompt. Per the §4D
  pre-registered reading: before any bigger run, fix RETRIEVAL ROUTING of playbook lessons into
  task-shaped turns (or accept that 1-lesson corpora can't mediate). Outcome transfer was
  structurally unmeasurable this run: ceiling × zero mediation.
- **(3) #27b frontier-vs-uniform: TIED AGAIN** — self-play yield per repeat treatment {1,0,1,1} vs
  uniform {1,0,1,1}: equal in all 4 repeats (B3: 2v2). Rule-as-written (≥ in ≥2/3) technically
  KEEPS frontier on ties, but two instrumented experiments now show zero separation — parsimony
  argues flipping the default to uniform (--no-frontier-selfplay) and shelving the frontier
  selection layer. OPERATOR DECISION pending; PRM stays either way (self-play fired 3/4 repeats
  per arm).
- **(4) Dream gate DEFINITIVELY adjudicated:** auto_memories=0 across all 12 arm-runs — the
  smart-memory consolidator processed the seeding turns (journal drains ran) but stored ZERO
  auto-facts, because task-shaped turns ("read sales.csv, compute X") contain nothing scoring ≥0.9
  as a memorable fact. The §4D "--smart-memory feeds the gate" hypothesis is REFUTED for task
  seeding; fact-shaped chat seeds (B3) don't fire it either. The 2026-07-07 hypothesis is now the
  only live path: **dream needs a trajectory-shaped seed source (code change), not a better
  seeding protocol (more runs).** Also: failed_traj=0 — the model passed even the seed_hard tasks,
  so reflection had no material (reflection yield 0 everywhere, consistent).
- Also observed: a `perfection_protocol` lesson source (Perfect-It internal learning) fired 1-2×
  in several arms — balanced across frontier/uniform, first time it shows in an ablation.

### 2026-07-09 (night) — B4 pilot #1: harness bug caught + battery recalibrated (pilot #2 running)
- **Pilot #1 ran clean** (prod stopped by operator; 3 passes × 23 candidates, 12:38-14:40) and
  earned its keep twice:
- **(1) Timeout-bleed cascade (REAL harness bug, would have poisoned the overnight run).** A
  client-side probe timeout does NOT stop the agent's in-flight turn — the agent keeps working it
  (artifact appeared on disk minutes after the driver moved on), while the next probe queues behind
  the #22 turn-serialization semaphore and burns its own 300s budget waiting. `conc_worker_sum`
  (model writes deadlock-prone producer/consumer code, genuine 300s overrun ×3) took down
  `web_table_sum` (0/3, never actually measured) and `web_pdf_links` (0/3 until the cascade drained
  in pass 3) — confirmed by per-pass duration records (300/300/300 and 300/300/164). Bonus finding:
  both web tasks shared the fixture filename `page.html`, so the overrunning task saw its file
  swapped mid-flight ("The file has no table structure"). **Fixes:** driver `_wait_arm_quiet`
  (poll the arm log's Request-Finished count vs requests sent, bounded grace, between every task);
  globally-unique fixture filenames across ALL tasks (`_rename_fixture`/`_rekey_expected` wrappers)
  + a uniqueness test.
- **(2) Calibration verdict: clean single-file tasks are CEILING** — every sql/algo/pg task and
  most bash/rp went 3/3 fast (42-91s); the model is stronger than the template-bank shapes at this
  tier. In-band (7/23): all four data_analysis tasks + rp_5xx_count + bash_top_user (~2/3 each) —
  the working difficulty lever is **messy multi-file data + fiddly-but-precise rules**, not
  algorithmic complexity. Survivors gave ZERO weak-cluster coverage (sql/algo/conc), which #27b
  needs. **12 v2 variants authored** porting the messy-data recipe into the ceiling clusters
  (dirty joins in SQL, 3-table payout, interval gaps with boundary rules, second-mode with
  tie-breaks, multi-condition log parsing, nested JSONL, ThreadPoolExecutor-named concurrency…).
  Battery tests 102 green (self-consistency gate covers all 35 candidates). **Pilot #2** (12 new +
  3 raced tasks, fixed driver, `--battery-file` subset support added to pilot mode) launched
  ~15:3x; final battery = pilot-1 survivors + pilot-2 survivors.
- **Pilot #2 (completed 17:57 after 3 harness iterations of its own):** (a) v2 crawled — my
  wait-for-quiet counted "Request Finished" (title case) but the pretty-stream renders it
  lowercase → every task burned the full 240s grace; fixed case-insensitive. (b) `conc_worker_sum`
  wedged the arm 37+ min on its pass-2 seed (deadlocking queue code retried across turns, exit 124
  each time) and the grace-then-PROCEED design re-created the cascade → task DROPPED as a
  run-killer (two pilots of evidence) and wait-for-quiet hardened to grace-then-HOLD (900s grace,
  1800s ceiling; a probe fired into a busy arm is a wasted probe). (c) A test I wrote for the old
  proceed behavior then silently hung the relaunch chain — hold ceiling parameterized, test split
  into case-insensitivity + hold-then-give-up (103 green). **Clean re-pilot verdict: the v2
  "harder" tasks mostly ALSO went ceiling** (9/12 at 3/3 — the model absorbs messy single-concept
  tasks too); survivors: algo_second_mode 2/3, web_table_sum 1/3 (its pilot-1 0/3 was cascade
  contamination; conversely web_pdf_links' pilot-1 "1/3" was contamination — clean 3/3, dropped).
  `sql_eng_payout`'s one fail was `7646.0` vs `7646` — a float-formatting artifact, so the verifier
  now normalizes integer-valued floats and the task reclassifies as ceiling.
- **FINAL BATTERY (`ablation_out/b4_battery_final.json`): 8 tasks** — da×4 (group_sum, join_gold,
  revenue, top_region), bash_top_user, rp_5xx_count, algo_second_mode, web_table_sum. Honest power
  note: at 8 tasks the stratified test only detects LARGE pass-rate effects; the run's mediation /
  dream-gate / lesson-yield instrumentation is fully informative regardless. **Pre-registration
  amendment (BEFORE the run):** weak-cluster in-band coverage ended at algo×1 + regex×1 (sql and
  concurrency calibrated out entirely), so #27b's "weak-cluster probe delta ≥ 0" condition is
  under-powered to meaninglessness — the frontier keep/flip verdict falls back to the YIELD
  criterion alone (self-play yield ≥ uniform in ≥2/3 repeats), with the delta reported
  descriptively only.

### 2026-07-09 (night) — #5 step 3 SHIPPED (finalization chain → _finalize_and_return)
- Same script-driven protocol as step 2 (content boundary asserts, dedent-safety, ast.parse +
  symtable gates). This region was STRUCTURALLY simpler than the dispatch pipeline: it is the tail
  of handle_chat inside the semaphore `async with` — single return, no nonlocal, no except handler
  between region and function end (just the Request-Finished `finally`), and nothing after it reads
  locals → **zero control-flow rewrites and zero repack** (`FinalizeState` = 20 read-only fields).
  Capture-analysis nuances this time: except-`as` bindings and in-function `import`s masquerade as
  unbound loads (same as step 2); `payload` was only bound inside the turn loop, so a `payload =
  None` pre-bind was added for the deterministic-dispatch exit path (crash semantics preserved:
  that theoretical path now AttributeErrors instead of NameErrors). handle_chat shrinks ~950 more
  lines (steps 1-3 total: ~2,900 of the original 11k). Suite **6801 green**; +6 direct tests;
  3 stale source-inspection tests repointed (one had a second latent staleness: it matched the
  bare literal `<tool_response>` that only existed in an unrelated comment). Live-validated on a
  throwaway agent (:8046, same code): probe file exact-bytes on disk, verifier gate (inside the
  extracted chain) CONFIRMED 100%, zero errors in the log. Prod deploys on next restart.

### 2026-07-09 (later) — B4 battery IMPLEMENTED + #5 step 2 SHIPPED (dispatch pipeline + TurnState)
- **B4 outcome battery implemented** (§4D design → code, same day): `scripts/trackb4_tasks.py` — 22
  probe candidates (7 seeded clusters + held-out web_automation far ring; unique-winner fixture
  post-processing kills tie ambiguity) + 8 seeding tasks (4 easy strong-cluster / 4 hard on the
  pre-registered weak clusters); every task self-consistency-gated (reference must verify, garbage
  must not). `scripts/ablation_trackb4.py` — seeding phase, per-probe mediation capture (playbook
  retrieval-credit diffs), task-stratified sign-flip test beside McNemar, log-based dream-gate
  instrumentation, `--pilot` calibration emitting `b4_battery.json`, `--smart-memory 0.9` in every
  arm, per-repeat fixture seeds (memorisation guard) identical across arms (pairing). +74 tests
  (`test_trackb4_battery.py` — caught a real tokenizer bug: sentence-final `25.` ≠ `25`). Docs:
  `scripts/ABLATION.md` §Track B4. Remaining: operator runs pilot (~2 h) + overnight (~11 h).
- **#5 step 2 shipped:** `_dispatch_and_process_tool_batch` + `TurnState` (see §4A #5 for the full
  contract). Method: script-driven surgery with content boundary asserts, dedent-safety check,
  `ast.parse` + symtable free-name gates; the AST capture analysis found the naive contract would
  have silently reset the cross-iteration SYSTEM-3 latch (`_request_sys3_fired_once`) — the exact
  failure class the decomposition memory warned about. Also: the old `break`'s "exit the
  enumerate(results) loop" comment was STALE — AST proves it broke the TURN loop.
- Suite **6795 passed / 12 skipped / 0 failed** (+80 today). **Deployed + live-validated** (operator
  restarted prod): (1) file-write+read request ran through `_dispatch_and_process_tool_batch` — the
  native tool_call corruption repair fired and recovered live, verifier CONFIRMED 100%, exact bytes
  on disk; (2) the model guessed `workspace action="search"` exactly as the §4C finding predicted
  and got a real search + the no-match redirect instead of a strike (its "no record of step2_check"
  answer is CORRECT — file_system writes don't record activity events; only commands/research/
  notes/tracked files do). Today's earlier fixes (stamping ContextVar, journal re-queue) are live
  in the same deploy.

### 2026-07-09 — §4 sweep: stamping race actually closed, workspace search, journal re-queue, B4 design
- **Event-stamping race (§4B): the "confirm #22 closes it" pass DISCONFIRMED it.** Two project-id
  fields exist; #22 + the 2026-07-08 pinning protect `context.current_project_id` (sandbox scoping)
  while every `record_*` stamps from `workspace_model.current_project_id`. Idle autoadvance was
  mis-stamping deterministically; dream self-play (own semaphore) clobbered the shared field and
  polluted the real activity log. Fixed with a task-local ContextVar override
  (`set_event_project` / `pinned_event_project` in `workspace/model.py`) read first by every stamp
  site, + `isolated_context.workspace_model = None` for self-play. +8 tests incl. the reproduced
  interleave.
- **`workspace` search action (§4C):** `action="search"` (alias `recall`) is now real —
  IDF-weighted keyword search over the activity log (`WorkspaceActivity.search`, `search_my_past`
  sibling); schema advertises `search`+`query`; near-miss arg names absorbed; no-match reply
  redirects to `recall`/`manage_projects`; enum⊆dispatch consistency test. +14 tests.
- **smart-memory 503 (§4C):** verified HTTP-layer retry exists but final failure silently+permanently
  dropped the popped journal item (and timeouts got no retry at all). Added task-level bounded
  re-queue: `RetryableConsolidationError` + `is_upstream_transient` (5xx/timeout/conn) in
  `memory/journal.py`, drain loop re-queues with `retries` cap 2, visible WARNINGs. +8 tests.
- **B4 outcome-battery DESIGNED (§4D)** for #4/#27b: behavioral-style grounded tasks over the
  self-play cluster families in three transfer rings, a [0.3,0.7] calibration pilot (the ceiling
  fix), an identical-in-all-arms seeding phase (real failures → reflection; `--smart-memory 0.9` →
  dream's `type:"auto"` gate — B3's arms never passed the flag and the consolidation task is the
  only auto-fragment writer, so dream was unsatisfiable by construction; cluster variance → frontier
  signal), per-probe lesson-retrieval mediation instrumentation, task-stratified stats at 3 repeats,
  and a pre-registered frontier keep/flip rule. Implementation = next focused session.
- Suite: 6685→**6715 passed** (+30), 12 skipped, 0 failed. Docs: tools/workspace.html,
  core/workspace_model.html, core/project_advancer.html, core/dream.html, memory/journal.html.
  **Deploy note: prod needs a restart to pick up the fixes.**

### 2026-07-08 (night) — deep_research per-URL Tor fetch racing
- Last leg of the Tor pipeline: the page-FETCH stage shared ONE circuit across all 8 URLs (the same
  correlation flaw fixed in search), the outer `wait_for` was 15s < the client's 20s (guillotined
  slow-but-live fetches — the mojeek-timeout twin), and there was no retry. **Measured before/after
  over live Tor (8 real URLs): 6/8 in 21s → 8/8 in 14s** (both MISSes recovered: one via its own fresh
  exit in 2s, one via timeout headroom). Fix in `tool_deep_research`: per-URL circuit
  (`_proxy_for_attempt(url, attempt, salt="fetch")`), retry circuit-retryable failures
  (timeout/503/conn/5xx via `_fetch_error_is_retryable`) on a fresh exit, skip definitive ones
  (binary/401/403/SSRF/4xx), outer timeout 22s > client 20s, sem 2→3, NEWNYM still suppressed. +16 tests
  (`tests/test_deep_research_fetch_racing.py`), docs + Tor memory updated. Test gotcha logged: dual
  module-name patch target + AsyncMock sync-side_effect.

### 2026-07-08 (night) — verifier-log clarity + host-process blind spot
- **LATE-verdict-empty differentiated.** Traced the `None` paths: no substantive evidence tool
  (bookkeeping-only turn), no verifier/llm_client attached (sim/ablation), empty final content, or
  strict-trivial-chat. So the sim-turn firing WAS by design — but one ambiguous line covered all of
  them AND the case it exists for (a dead verifier path). `_record_late_verdict` now takes `last_tool`
  and emits 3 distinct messages: no-evidence → INFO by-design; verifier-not-attached → INFO by-design;
  evidence + verifier present yet no verdict → **WARNING** (trivial-chat skip or real error). The
  warning is now rare and meaningful.
- **Host-process blind spot closed at the tool level.** The sandbox has its own PID namespace: a
  `pkill -f app.py` aimed at the USER's host-run server exits 0 and kills NOTHING — no error text for
  the fallback-hint table to match, and the model concludes it restarted the server (chess session,
  twice). `execute` now detects name-based kills (`pkill`/`killall`/`kill $(pgrep …)`) and appends
  ground truth to BOTH outcomes (the "success" is the dangerous one), naming the right action: tell the
  user to restart it. `execute` schema warns up front. Same philosophy as the loopback guard: the exit
  code lies, so the tool tells the truth.
- Tests: +11 (`tests/test_host_process_and_verdict_clarity.py`) — 3 verdict branches, kill-pattern
  detector, note contents, schema warning. Docs: agent.html, execute.html. Operator ran the suite:
  all green. Deployed (server restarted).

### 2026-07-08 (night) — guard-box incident fixed (request 04 post-mortem)
- **Incident:** resume-Chess-Coach request got boxed in with ZERO legal write paths for ~6 min: two
  content-less `replace` calls seeded the pre-flight guard, whose (tool,target) key then blocked the
  CORRECT replace→write recovery 3× (each after ~80 s of full-file generation); the escape via execute
  heredoc was blocked by the egress guard because the FILE CONTENT legitimately mentions
  127.0.0.1:8000 (the chess app calls Ghost by design). Guard blocks advanced no loop budget.
- **Fix 1 — pre-flight guard:** key now `(tool, target, operation)` (`RecentFailureGuard.record/
  would_repeat` gained `op`; dispatch + record call sites thread `operation|action`); block message
  names LEGAL alternatives; per-request block budget — 2 guard blocks force a final reply
  (state attempts + exact error + ask the user).
- **Fix 2 — egress guard:** shell commands judged by `_command_probes_agent_port` — heredoc bodies
  stripped (data, not execution) AND a network-client token (curl/wget/nc/requests/urllib/httpx/…)
  must co-occur with the loopback URL. Direct probes still blocked (incl. after a heredoc); text
  manipulation (`echo … > file`) allowed. Inline `content` keeps the strict rule (executed code).
  Guard message now names the legal file-writing path.
- **Fix 3 — replace steer:** content-less `replace` error now names the escape (`operation='write'`
  with the full file, don't retry replace). Raw args unavailable (trajectory not flushed) so
  model-slip vs native-args corruption unconfirmed; steer helps either way.
- Tests: +11 (`tests/test_guard_box_fixes.py`) incl. the exact false-positive heredoc and the
  replace→write recovery. Docs: execute.html, agent.html, file_system.html. Note: agent eventually
  escaped on its own (~17:00) and appended save/load endpoints to app.py.

### 2026-07-08 (late) — chess-eval improvements #1-#3 SHIPPED (verification reflexes)
- **(1) Constraint gate** (`core/build_gates.constraint_gate`): one background LLM audit of the files a
  task produced vs the project's stored constraints — JUDGMENT-based, closing the gap the evidence-based
  DONE-gate left (model wrote compliant prose while shipping a forbidden engine). Wired into BOTH paths:
  `build_coding_task` (violation = retry feedback quoting constraint+evidence) and interactive
  `task_update→DONE` (refusal + `CONSTRAINT-OVERRIDE:` escape hatch for user-approved exceptions).
  Fails open on infra errors.
- **(2) Smoke gate** (`core/build_gates.smoke_gate`): after the spec verify, `py_compile` every written
  .py + Flask route sweep via `test_client` (GET must not 5xx; POST gets `{}`, may 4xx not 5xx);
  SIGALRM(45s) self-bound so a blocking-server import can't wedge the exec. Wired into
  `build_coding_task`; failures feed the retry loop.
- **(3) Probe-before-hypothesis loop breaker**: no-progress trip now at the 2nd identical read (was 3),
  hard abort at 3 (was 5); all steer/abort messages rewritten to lead with EVIDENCE-GATHERING (probe the
  URL/command, apply the change, or ask the user for the exact error from devtools) instead of only
  "trust what you have". The chess session's 536 s / 5-re-read spiral is the motivating trace.
- Tests: +18 (`test_build_gates.py` incl. real smoke script vs a 500-ing Flask app;
  `test_probe_before_hypothesis.py`; happy-path executor contract updated). Docs: coding_executor.html,
  agent.html, projects.html. Suite 6644 green (2 known FORCE_COLOR env-flakes only). Deploy: needs
  restart.

### 2026-07-08 (late) — chess project fixed by operator-side Claude; weakness eval from the session
- **Chess Coach (30d5d5b65c38) fixed & validated end-to-end** (state/move/illegal/undo/reset; Ghost
  answered e2e4 with e7e5 + in-voice coaching via a REAL agent call). What was broken: `get_ghost_move`
  was a random/heuristic engine (violating the stored "Ghost plays directly" constraint the model itself
  restated in its plan); 3 crash bugs (`board.move_stack < 8` list-vs-int TypeError, `random` scoped to
  another function → NameError, nonexistent `board.move_is_check`); frontend rendered the analysis dict
  as `[object Object]`, expected fields the backend never sent, wiped history via `game.load(fen)`,
  client-only undo desynced the server, dead appended script block; pieceTheme flip-flopped between two
  CDNs — measured: chessboardjs.com=200, unpkg img path=404 (agent never probed either). Rebuilt:
  backend asks the agent (`/api/chat`) for Black's move + comment with strict-JSON contract, legal-move
  validation, one corrective retry, honest 502 (NO engine fallback, per constraint), server-side undo,
  history-as-source-of-truth; frontend contract aligned. Project game_state reset.
- **Agent weaknesses observed (8 chess requests, candidates for §4):** (1) constraints inform but don't
  GATE — model restated the no-engine constraint, then built an engine anyway; needs a post-build
  constraint check against the diff (sibling of the self-play reference gate). (2) No endpoint smoke-run
  after codegen — 3 crash bugs shipped; one curl per route would catch them. (3) Debugging by hypothesis
  looping instead of probing — E8 spun 536s/18 turns re-reading the same file 5× (loop breaker saved it),
  flip-flopped the CDN URL twice without ever curling it, never asked the user for the failing URL from
  devtools. (4) Host-vs-sandbox process model gap — tried twice to pkill/restart the USER's host-run
  Flask from inside the sandbox instead of telling the user "restart app.py to pick up the fix" (same
  family as the loopback blind spot). (5) Cross-file contract drift within one session — backend/frontend
  it authored hours apart disagree on the response schema; coding tasks should grep consumers of a
  changed endpoint. Positives: constraint replay fired, native-tools repair ×2, loop breaker fired,
  late verifier REFUTED a non-answer and its correction surfaced next turn.

### 2026-07-08 (evening) — thinking visible in non-verbose logs
- Operator request: non-verbose launches truncate nicely but 💭 thinking was filtered out entirely.
  Root cause: `_emit_thinking`/`_flush_thinking` (streaming closures) returned early unless
  VERBOSE_MODE — thinking never reached pretty_log at all. Fix (two rounds, operator-directed):
  gates removed AND thinking exempted from the content budget — `pretty_log` gained a per-call
  `no_truncate` flag, passed at the 3 thinking call sites, so the FULL reasoning stream is visible in
  every mode (newline-flatten/redact/wrap still apply → identical line format); all other lines keep
  the standard 60-char budget in non-verbose; 🧠 post-stream summary unchanged. +6 tests
  (`tests/test_logging_thinking_nonverbose.py`), docs/logging.html updated.
  #8 step 1 (parser extraction) closed as DONE on the task ledger; steps 2-4 remain journaled below
  and in the agent-py-decomposition memory for a focused session.

### 2026-07-08 (evening) — #8 agent.py decomposition, step 1 of 4 SHIPPED
- **`_parse_assistant_tool_calls` extracted** from handle_chat: the ~640-line robust tool-call parser
  (XML normalization heals, truncation detector, flood cap, native tool_calls corruption repair,
  raw-JSON fallback) is now a method with contract
  `(content, msg) → (tool_calls, ui_content, parse_failure_reason)`. Extraction was VERBATIM
  (script-driven dedent, ast.parse gate, boundary asserts) — verified no return/await/loop-control
  crossed the boundary; only `content`, `msg`, `self.available_tools` do. Caller keeps think-strip,
  leak scrubbers, history assignment. agent.py: handle_chat shrinks ~640 lines.
- **Validation:** full suite 6614 green post-extraction; **+9 direct unit tests**
  (`tests/test_parse_assistant_tool_calls.py` — first-ever isolated coverage of this hot path: XML
  canonical/bare-function/sloppy-attrs, think-block immunity, truncation flag+recovery, native
  pass-through, raw-JSON recovery); **LIVE hot-path validated** (agent restarted; request EB drove a
  real file_system tool call through the new method on the native path — verifier CONFIRMED 100%,
  zero parse errors).
- **Remaining steps (next focused session):** (2) tool guard/dispatch/result pipeline, (3) finalization
  chain, (4) final-generation streamer closure + `TurnState` dataclass. Same protocol: verbatim
  extraction → suite → restart → live turn before the next step.

### 2026-07-08 (later) — Full-day log eval → 3 defects fixed
- **Log evaluation (12 requests since morning restart):** search stack 0 strike-outs all day (racing +
  terse logs working); async verifier LATE-REFUTED request 67 (model listed 12 PG19 features from the
  devel "fill in later" skeleton page — fabrication caught, correction queued); native-tools guard fired
  again; PRM/router/calib/reflection/autoadvance all closed loops in idle. Three defects found & fixed:
- **(1) frontier.py `record_run` KeyError 'runs':** `note_reflection_failure` created clusters as bare
  `{}`; record_run's full-defaults setdefault no-ops on existing dicts → `cluster["runs"]` raised and NO
  self-play run was ever recorded for `python_general` (live state file confirmed: only
  `reflection_failures` key). Fix: `_ensure_cluster` = setdefault + per-key back-fill from
  `_cluster_defaults()`, used by both writers; state file heals on next write. +3 tests
  (test_frontier_tracker.py::TestPartialClusterSchemaBackfill).
- **(2) self-play unwinnable-challenge gate:** LLM-generated critical-path challenge had validator
  expecting duration=10 while its own tasks.json yields 25 — echo self-test can't catch it (validator
  doesn't crash, it's just wrong about the data); solver failed 3/3 on CORRECT code, cluster `algo` got
  a bogus -1.0 delta + misleading lesson. Fix: challenges must emit `<reference_solution>` (computes
  answer FROM setup files); static gate `validate_reference_solution` rejects hardcoded references
  (must open a setup file), sandbox gate runs reference → validator against real data and DISCARDS the
  challenge on any non-zero exit. Templates/journal challenges skip (pre-verified); omitted block =
  logged warning, gate skipped. +8 tests (test_selfplay_reference_gate.py).
- **(3) autoadvance workspace placement:** idle ticks carry no conversation → process-global
  `current_project_id` parked → TinyAI's model/train/evaluate files written to sandbox ROOT; interactive
  session after `switch` saw only `projects/<id>/` and recreated the demo from scratch. Fix:
  `pinned_project_context(context, project_id)` proxy (pins the id, delegates reads/writes to base);
  BOTH tool-runner sites (agent.py idle phase 2.95 + manage_projects autoadvance batch) build tools from
  it. Also fixes the mid-batch reconcile race. +7 tests (test_autoadvance_project_scope.py).
- Docs: memory/frontier.html (schema back-fill), core/dream.html (gate 4), core/project_advancer.html
  (workspace pinning). NOTE: existing root-level TinyAI artifacts were NOT moved — new builds land in
  the project dir; the stray root files remain for the operator to reconcile.

### 2026-07-08 — Tor web-search: per-engine circuit race (search.py)
- **Trigger:** operator's live session showed searches failing wholesale ("ZERO results across all
  engines and circuits", 74 s burned) and the model then hallucinating "research results" from its own
  knowledge and marking the research task DONE.
- **Measured first** (42 probes: 7 engines × 2 queries × 3 fresh circuits): per-(engine,circuit)
  success **~10%** — brave 2/6, yahoo 1/6, mojeek 1/6, rest 0/6; even `python asyncio tutorial` got
  1/21. Failures are exit-IP-driven, per my standing note (measure before touching the engine set).
- **Root cause:** ddgs multi-backend mode runs all engines through the ONE proxy on the DDGS instance
  → all engines shared a circuit per attempt → **correlated failure** (~10%, not breadth). Bonus: with
  max_results=20, ddgs caps internal fan-out at 3 engine workers, so "5-engine breadth" was ~3-on-1-exit.
- **Fix (`_race_search_wave`):** one single-engine ddgs call PER ENGINE, each on its OWN circuit
  (per-engine salt in the SOCKS auth tag), first non-empty junk-filtered result wins, losers cancelled;
  wave deadline = ddgs timeout + grace. web_search: 2 waves + 2 reformulation waves (~24 independent
  tickets ≈ 90%+); deep_research: 2 waves. `yahoo` re-added (re-measured: fails fast ~1.4-2.2 s, no
  longer hangs, won a probe); `grokipedia` stays out (typeahead API, 0/6). StopIteration from an engine
  is converted to RuntimeError (PEP 479 would poison asyncio future chaining — surfaced by the suite).
- **Live-validated over real Tor: 3/3 queries** (incl. both exact failing prod queries) **won wave 0 in
  ~2 s** (yahoo ×2, duckduckgo ×1) vs 74 s → zero before. Tests: +4 `test_search_engine_race.py` (race
  semantics) + updated contracts in hardening/resilience/enhancements/audit suites (nondeterministic
  call order → query-keyed mocks; cache isolation). Docs: `docs/tools/search.html` rewritten (race
  section). **Deploy note: prod needs a restart to pick this up.**
- Residual (not code): the model *inventing* sources after a failed search is a separate honesty issue —
  the error text already instructs "state that web search was unavailable"; watch whether the faster
  successful path makes this moot.
- **Post-restart follow-up (same day):** live tally after deploy: 9 searches → 5 won wave 0/1 (~2 s),
  3 saved by reformulation waves, 1 total strike-out — an 11-word keyword-stuffed query (hallucinated
  "postgresql 20"); residual failures are query-side. Three refinements shipped: (1) wave-failure log
  groups engines by identical error + strips URLs (`_brief_engine_error` — fixes the cryptic `url (h`
  truncation); (2) every wave log line carries a ‹query› tag (concurrent searches interleave — a
  "won wave 10" next to "ZERO results" was two different searches); (3) `_reformulate_query` hard-trims
  >6-word queries to the first 5 words of the broadened form instead of "how to {query}" (which kept
  every rare term and failed identically). Tests: +3 (race log grouping/tagging, reformulation trim).
- **Second live pass (post-restart eval + operator log feedback):** request 68 (PG20 deep research) went
  10/10 searches, 0 strike-outs (5 wave-0/1 wins ~1-4 s, 4 reformulation rescues incl. the new hard-trim;
  yahoo won 4 races). Operator asked for terser failure lines → wave failures now ONE categorized line
  (`no winner — 5 empty; mojeek conn-error` via `_failure_category`; unknown errors keep a 48-char
  snippet; full sanitized detail → logger.debug); URL-strip regex keeps closing punctuation. Correction:
  "postgresql 20" is a real in-dev version (cycle opened 2026-06-29), so the earlier strike-out was pure
  keyword stuffing. Known-slow residual: deep_research per-URL page fetches over Tor.

### 2026-07-07 — Correctness/security sweep + deeper B3 + #6/#7 closed
- **7 headless fixes** shipped with tests + docs, suite 6528→**6587** green (0 fail): graph-compression
  expired-fact resurrection (temporal-safe node merge, +7); correction-lookup banner fingerprint
  (`_strip_leading_banners`, +11 incl. e2e promotion gate); browser SSRF residuals (`file://` subtree +
  non-Tor DNS-rebind re-resolve, +24); vector smart_update subject-key guard + correct_fragment upsert
  (+8); projects/skills cross-process locks (`BEGIN IMMEDIATE` + fcntl/PID-temp, +7); PRM mode-aware
  training-viability gate + serve-skew warning (+6). Plus 2 stale tests updated (PID-unique tmp; SSRF
  probe marker tolerant of new kwargs). §4B entries marked RESOLVED.
- **#6 (pin) CLOSED** — durable already via launcher; code-default flip tried and reverted (broke 8
  prompt-assembly tests). **#7 (schema diet) CLOSED — accept lean** (operator): descriptions already
  lean; residual size is necessary param-contract; KV pin amortizes; blind trim is net-negative risk.
- **Deeper B3 (#4/#27b) executed.** Extended the harness: 18 enriched seeds (was 10 thin fact-recalls),
  a 3rd `treatment_uniform` (`--no-frontier-selfplay`) arm, exact-McNemar + frontier-yield in the
  report. 3 arms × 2 repeats vs the shared llama (prod stopped for RAM). Idle loops productive (treat 4
  / uniform 2 / control 0 lessons; reflection fired). BUT: recall-probe McNemar is a **ceiling artifact**
  (97% both arms) — wrong instrument; frontier vs uniform self-play **tied 2v2** (inconclusive); dream
  still didn't fire on fact-seeds. Report: `ablation_out/trackb3-20260707-191216/`. Open work (§4A):
  harder task battery + more repeats. **#8 (agent.py refactor) deferred** to a focused session by operator.

### 2026-07-07 — Live validations after the improvement review
- **KV pin (#6) validated in prod.** Across the 2 turns of a real coding request, the
  `stable-prefix h=a449281b len=26116` log line was byte-identical → the pin holds and #7c
  (byte-stable tool set) works; no re-prefill between turns. `GHOST_PIN_TOOL_SCHEMAS=1` set by
  operator.
- **B3 first live pass (#4).** Operator stopped prod (disabled the launchd job) to free RAM;
  isolated treatment/control agents booted on :8046 against the shared llama :8088. Treatment
  (`--bio-time-scale 30 --bio-deterministic`) fired reflection + synthetic self-play within minutes
  of accelerated idle and produced **1 `self_play` lesson** ("alternative idiom for sql tasks",
  challenge SOLVED, score +1.900); control (scale 1) produced **0**. First live proof a pure-idle
  loop is productive. Dream didn't fire (too few seed memories for entropy); reflection had no failed
  turns. Harness completed both arms; a report-builder bug (reused trackb2's `_build_report` with
  wrong meta keys) crashed the final formatting — data recovered from the temp GHOST_HOMEs, and
  `_b3_report` written to fix it. Prod restored via the venv-python restart (the bare homebrew python
  lacks uvicorn).

### 2026-07-07 — 6-agent improvement review + implementation (24/27 done)
A 6-agent parallel review (core loop / tools / memory / learning loops / service layer / cross-cut)
produced 27 high-impact items. Implemented 24 headless with tests + docs; the rest need live sessions
(§4A). Full suite 6398→6508 (+110 tests). Highlights: the 600s inline-`is_background` self-stall
(#2/#19, same root cause as the fixed critic deadlock); RSS watchdog + `/api/health` + resolved-config
dump (#3/#21); the MemoryBus read-path rebuild (#13/#14/#15/#16 — wired episodic semantic recall,
per-item RRF, tier-scoring, deferred retrieval credit); de-quadratic streaming (#12); one truncation
policy + spill-to-file (#10); turn serialization (#22); `spawn_bg` unification (#20); context_manager
wiring (#27a); graph forgetting (#27c); reconstructed the lost redesign doc (#25); the guard-module
seam (#5, partial). Per-item detail in the Log table (preserved in git history).

### 2026-07-05 — Post-sweep live regressions (from the operator's real sessions)
- **skills_auto graduation producer wiring** (BUGHUNT unit-25 deferred, closed). The pipeline was
  structurally unreachable in prod: async-critic records the trajectory UNKNOWN before the verdict
  lands, and nothing backfilled the corpus (2058 UNKNOWN, 0 extractor-eligible). Fixed:
  `_record_late_verdict → _backfill_trajectory_outcome` (CONFIRMED≥0.7→PASSED guarded UNKNOWN-only;
  REFUTED≥0.7→FAILED + Reflector/PRM negative; corrections sidecar wins). ROOT CAUSE of "no late
  verdict ever": `--no-verifier` had been in the launcher (unintended) — a SIGUSR2 task-dump proved
  the verdict task was never spawned; permanently REMOVED (postmortem-off stays deliberate).
  LIVE-VERIFIED end-to-end.
- **Deferred closures:** router matmul warning noise (Apple Accelerate FPE on a provably-finite
  matmul outside `errstate` — wrapped + `_vectorize` choke-point sanitisation); WEB-EXEC fail-open
  (an inconclusive probe on a web-artifact write now caps CONFIRMED at 0.6 + fires the "actually RUN
  it" repair); correction-classifier affirmation false-positive (praise opening "actually" no longer
  promotes-to-FAILED — affirmation veto).
- **The chess post-mortem (5 fixes).** "play chess against each other, with YOU" failed totally: the
  LLM wrote 5 crash bugs + 1 comprehension inversion, and the agent had every signal to stop and used
  none. Fixed: async-critic mode shipped untested writes (the in-loop repair was gated behind
  `not async` — the pure unverified-mutation predicate now fires inline; this is the same gap #18
  later fully closed); project constraints never reached the autoadvance coding executor; participant-
  mode architecture steer + `/api/game/move`; poisoned memories + no surgical delete (added
  `VectorMemory.delete_fragment` + `POST /api/memory/delete`); `update_profile` couldn't delete +
  idempotency hash poisoned by failures (commit at success, not dispatch).

### 2026-07-04 — Functional bug hunt (14 units) + post-hunt strategy
- All 14 live subsystems driven with real requests → CLEAR. Real fixes: turn-loop `<think>`-strip cut
  at a quoted `<tool_call>` mention (unit 1); **insert_fact hung the turn + dropped the fact** — the
  bus path awaited graph-triplet extraction INLINE with `is_background=True` → self-deadlock (the
  original of the #2 class; fixed by publish-then-extract-in-background, unit 3); native tool-call
  corruption repair (units 5/7 — 3 variants: the upstream server leaks following calls' XML into the
  first arg under `--native-tools`); MCTS value fn returned flat 0.50 because the sim call's 256-token
  budget was consumed by `<think>` → disabled thinking on the sim call (unit 11).
- **Post-hunt strategy (make success measurable):** built an offline invariant GATE
  (`eval_baseline gate`), consolidated the ~5 disagreeing "was this turn good?" signals into one
  grounded `resolve_turn_outcome` (verifier verdict now folds into the trajectory corpus, not just
  calibration/selfhood), persisted the reflection dedup set (loop now progresses through the backlog),
  and built an execution-grounded **behavioral** eval (`eval/behavioral.py` — drives the live agent +
  inspects its sandbox/DB/trajectory; `mean_tool_calls=2.4` vs capability's 0.0). Also a
  sandbox-verdict runner + discriminating behavioral eval (closed a recurring gap).

### 2026-07-03/04 — Static bug hunt (28 units)
- Every subsystem reviewed one-lens-per-unit (often several parallel review agents), findings
  re-verified in source before fixing. ~28 unit sessions, hundreds of findings, ~150 confirmed fixes
  + regression tests + HTML docs. Notable HIGH fixes: browser SSRF-on-redirect (in-sandbox route
  interceptor covering redirects + subresources + `.last_url` re-nav + Tor-bypass); destructive
  file_system ops resolving to the sandbox ROOT (allow_root=False guard); manage_projects delete
  hard-deleting the ACTIVE project (auto-fill yields to explicit title); project_advancer missing the
  EXIT-CODE banner (failed build marked DONE); dream leaking synthetic trajectories to the prod log;
  the competence-detector EXIT-CODE substring gap (codes 3-9/127/130 scored SUCCESS). The `**d`
  dataclass-from-dict schema-drift silent-drop/wipe pattern recurred across selfhood/distill/eval and
  was hardened everywhere. Residuals → §4B. Regression tests: `tests/test_bughunt_unit*.py`.
