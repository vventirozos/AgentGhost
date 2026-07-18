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

## 1. Current state (2026-07-11)

- **Four capability features shipped 2026-07-11** (§6) from a three-agent capability survey: the agent
  now (1) has a **mouth** — an autonomous-activity ledger feeding an all-phase next-turn digest + outbound
  push (webhook/ntfy/Slack), with scheduled-turn conclusions no longer discarded; (2) can **host** —
  supervised long-lived sandbox services it can drive with its own browser; (3) can **compose** —
  `save_as` data-flow between composed-skill steps, bounded tool-using sub-agent `delegate`, and a `jobs`
  status surface; (4) has **durable server-side sessions** + real turn cancellation that releases the
  global turn lock. Suite **7077 passed / 12 skipped / 0 failed**. *Prod needs a restart to pick these up.*
- The three surveyed gaps are now closed; the remaining open work in §4 is unchanged (GAIA run, #5 step 4,
  the #4/#27b outcome battery).

## 1b. Prior state (2026-07-07)

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
`/Library/LaunchDaemons/com.local.ghost-agent.plist` (**KeepAlive=true**). Live flags (2026-07-13):
`--verbose --deep-reason --smart-memory 0.9 --max-context 240000 --mandatory-tor
--autoadvance-idle --enable-metacog --metacog-mem-high 98 --metacog-mem-floor-mb 300
--visual-nodes http://127.0.0.1:8088|Eva --image-gen-nodes http://100.122.46.101:8000|Ghost
--worker-nodes http://100.83.184.117:8088|Nova`. (Off-main nodes use TAILNET IPs since
2026-07-17 — macOS Tahoe's Local Network privacy silently drops a system daemon's packets to
192.168.x before the wire; tailnet/utun is exempt. See §6 that date.)
Env: `GHOST_HOME=/Users/vasilis/Data/AI/Data/`, `GHOST_CRITIC_ASYNC=1`, `GHOST_CRITIC_NO_THINK=0`,
`GHOST_PIN_TOOL_SCHEMAS=1`, **`GHOST_API_KEY=$(cat ~/Data/AI/.ghost_api_key)`** (auth ENABLED
2026-07-13 — the launcher reads the canonical mode-600 secret file and exports the env var; no
`--api-key` argv so the secret stays out of `ps`. ALL API calls need `X-Ghost-Key`, /api/health
included). cwd `/Users/vasilis/Data/AI/Agent`. Verifier is ENABLED; postmortem is
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
export GHOST_API_KEY="$(cat /Users/vasilis/Data/AI/.ghost_api_key)"
/Users/vasilis/Data/AI/.agent.venv/bin/python -m src.ghost_agent.main --port 8000 \
  --upstream-url http://127.0.0.1:8088 --visual-nodes 'http://127.0.0.1:8088|Eva' \
  --image-gen-nodes 'http://192.168.0.155:8000|Ghost' --verbose --deep-reason \
  --smart-memory 0.9 --max-context 240000 --mandatory-tor --autoadvance-idle \
  --enable-metacog --metacog-mem-high 98 --metacog-mem-floor-mb 300 \
  >> /Users/vasilis/Data/AI/Logs/ghost-agent.log 2>&1 &
```
A manually-started prod is **unsupervised**; kill it before re-enabling the launchd service to
avoid a :8000 bind conflict.

**Drive a request** (model name validated — must be `qwen-3.6-35b-a3`; key required since 2026-07-13):
```bash
curl -s -m 180 -X POST http://127.0.0.1:8000/api/chat -H 'Content-Type: application/json' \
  -H "X-Ghost-Key: $(cat /Users/vasilis/Data/AI/.ghost_api_key)" \
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

**Reach an agent-hosted service remotely (2026-07-12).** An in-sandbox service is trapped behind
two walls. Wall 1 (automatic): it must run on a *published* port (`GHOST_SANDBOX_SERVICE_PORTS`,
default `8100-8104`) AND bind `0.0.0.0` inside the container — `services.py` `start()` now exports
`HOST=0.0.0.0`/`GHOST_SERVICE_HOST` next to `PORT`, so docker's bridge-publish forwards host
`127.0.0.1:<port>` → the app (a loopback-bound app is reachable in-sandbox but NOT from the host).
Wall 2 (operator, one command): published ports bind host loopback only (authless API on this
host), so expose one to the tailnet with `/Users/vasilis/Data/AI/bin/serve-remote.sh <port>` →
`https://eva.taila2b1d.ts.net:<port>/` (tailnet-only, real TLS; teardown `unserve-remote.sh
<port>`|`all`). `manage_services` surfaces the exact command + URL when a service comes up on a
published port. The agent can't run `tailscale serve` itself (sandboxed; exposure is an operator
action). See `docs/sandbox/services.html#remote-access`. **Needs a prod restart** (env in `start()`).

**Two fixes from the live chess-hosting functional test (2026-07-12).** (1) `manage_services`
accepted a `workdir` param end-to-end (handler → `start()` → `cd`) but it was MISSING from the
tool JSON schema, so the model couldn't see it and burned ~50s baking `cd` into the command
(tripping the loop-breaker) when hosting a subdirectory app — added `workdir` to the schema
`properties`. (2) Boot-only `warm_up_workers()` was insufficient: on a request whose worker idled
~105s during sandbox work, BOTH the front-of-request query expansion AND the finalize route
ReadTimeout'd at 5s — added `LLMClient.keepalive_workers()` (spawn_bg loop, pings each off-main
node every 45s, tunable via `GHOST_WORKER_KEEPALIVE_S`, ≤0 disables). The timeouts were harmless
to correctness but silently downgraded query expansion to legacy string-concat every request. Both
**need a prod restart**.

**Service manager leaked orphaned processes — root-caused + fixed (2026-07-12).** `manage_services`
launched with `setsid nohup sh cmd.sh & echo $!`, but `$!` was a TRANSIENT wrapper pid: under
`setsid` the real service (re-parented to the container's PID 1) had a different pid, so
`stop`/`restart` killed the wrong one and **orphaned the real process** (hung processes accumulated
across restarts; live evidence — registry said `chess-v4` pid=817 while `ss` showed the real
listener was pid=625). Fix: the generated `cmd.sh` now records its own pid (`echo $$ > <name>.pid`
as its first line; under setsid that shell is the session/group leader, so `kill -- -<pid>` reaps
the tree) and `start` registers THAT. Plus: a new `stop-all` action (stop every service + reclaim
ports + clear registry — the one-shot cleanup), a port-reclaim fallback in `stop` (kill the port's
listener via the now-baked `ss` when the tracked pid is dead but the port is still held), and a
`status` hint when dead entries exist. **Verified live 2026-07-12:** a restart cycle leaves exactly
ONE process (the old one reclaimed via the port fallback), no orphan. Refinement: the launched
`cmd.sh` now `exec`s a simple command so the recorded `$$` is the EXACT service pid (the shell
becomes it) rather than a wrapper the shell forks-then-exits (which left status() showing DEAD while
the service ran + made stop() rely on the port fallback). Compound commands can't be exec'd — they
keep the fallback. The exec refinement lands on the NEXT restart; the orphan fix itself is already
live.

**ZOMBIES were the deeper mechanism (2026-07-12, 137s-request postmortem).** Container PID 1 was
`sleep infinity` — it never wait()s, so every dead orphan became a PERMANENT zombie (`[sh]`, `[tor]`,
`[headless_shell]` `<defunct>` all accumulating), and **zombies pass `kill -0`** — so dead service
launchers looked "already running (pid N)", stop() no-op'd against them, and start()'s
exited-immediately+log-tail diagnostic never fired. Compounding it, the model passed
`workdir='/projects/<id>'` (missing /workspace) and the launch's `cd` failed INVISIBLY inside the
async subshell (the log redirection opens after the cd — nothing written anywhere) → 3 identical
failed launches → model worked around via `execute … &` (an unsupervised orphan served the chess
game). Fixes: (1) `run_kwargs["init"] = True` — tini as PID 1 reaps zombies (effective on container
recreate); (2) `_pid_alive` also rejects `/proc/<pid>/stat` state Z (works in old containers);
(3) start() VALIDATES workdir exists before launching, heals `/projects/…`→`/workspace/projects/…`,
anchors relative paths at /workspace, and strips a redundant `cd X &&` when workdir covers it.
**Needs prod restart; init needs a container recreate.**

**Keepalive log spam + hidden fallback bug (2026-07-12).** The 45s `keepalive → Worker Node (Nova)`
line spammed the live stream. Fix: heartbeats log TRANSITIONS, not ticks — healthy pings silent,
node-down = ONE warning, recovery = ONE line (per-ping traffic at debug, gated on
`task_label == "keepalive"`). Found underneath: keepalive didn't pass `off_main_only=True`, so every
failed ping FELL BACK TO THE MAIN 35B (max_tokens=1 on the single slot every 45s while a node was
down) — now raises `OffMainNodeUnavailable` instead, caught as the down-signal. Chess-app wonkiness
same day: move calls through /api/chat did tool gymnastics (bash-echo'd its JSON → quote-mangling →
verifier REFUTED repair; then WROTE chess_move.py to print JSON) at 37-57s/move — fixed app-side
(no-tools plain-text directive in the prompt + `X-Request-ID: sub-chess-…` marks moves internal,
suppressing the activity banner); verified 24s single-turn clean-JSON moves. **Keepalive fix needs a
prod restart; chess fix is live** (service restarted).

**4th ReadTimeout cause — internal requests loading nova (2026-07-12).** Chess-move (`sub-chess-`)
requests still ran the FULL memory pipeline: RAG-fusion DECOMPOSE_QUERY on the worker at request
start (critical path — the "+0.00s → 8s ReadTimeout" lines) and the smart-memory extract
(max_tokens=3072) at finalize — consecutive moves saturated nova so the next move's routing call hit
the ceiling. (Diagnosis was initially misdirected because route() hardcoded its log label as "query
expansion"; it was actually DECOMPOSE_QUERY — label now derives from the task.) Fix:
`is_internal_request(req_id)` gates hydration's `llm_client` (→ plain vector recall, no worker call)
and the smart-memory journal appends on both finalize paths (also stops chess FENs polluting
memory). **Needs prod restart.**

**Image generation — node + agent both fixed (2026-07-12).** ghost's Jetson image node had two
silent quality killers (77-token CLIP truncation ate the agent's long prompts; A1111 `(x:1.4)`
syntax entered CLIP as literal garbage) + an anime VAE (ClearVAE) overriding CyberRealistic's baked
realism VAE — fixed server-side (lpw chunked encoding + A1111 weight parser + baked VAE + DPM++ 2M
Karras pinned + seed/clip_skip params; verified by generating and LOOKING at before/after images;
see memory imggen-node-quality). Then the AGENT side turned out to still be tuned for the long-gone
DreamShaper LCM node: `image_gen.py` clamped steps 4-8 (schema literally said "Lightning models") —
server floor-raised to 15, HALF the tuned 30 — and snapped sizes to SDXL 1024²+ buckets that blow
the Jetson's 393k-px budget. Fixed: steps omitted by default (node default wins), SD1.5 bucket
ladder (512x768…768x512), seed/negative_prompt passthrough, schema/prompts now teach weight syntax
+ no-truncation, 503-aware image retry (8s for node warmup). **Agent side needs prod restart.**

**Sandbox image v5 + chess engine-opponent mode (2026-07-12).** Baked `stockfish` into the sandbox
image (docker.py apt + Dockerfile, marker `.v4`→`.v5`, base rebuilt; pinned tests bumped) for the
chess project's new engine mode. The chess side-project (app.py in the sandbox workspace) now lets
Black be Ghost (LLM, original "plays directly" mode) OR Stockfish at 8 difficulty levels — Ghost
always coaches: engine moves return instantly and the coach note streams async via
`/api/game/coach`, plus `/api/game/hint` (Ghost suggests the user's White move) in both modes. Flask
now `threaded=True` + `_state_lock`. Verified live: v5 recreate on prod restart brought PID 1 =
docker-init (tini/zombie fix now LIVE too) + stockfish baked; engine replies instant, coach gives
real Alekhine coaching, all endpoints 200 in the user's live session.

**Chat→project promotion nudge retuned (2026-07-12).** The "💡 promote to a tracked project" footer
(agent.py finalize → `project_safety.should_suggest_promotion`) fired on 12 turns of pure chat and
titled the project "hello" (the first user turn). Fixes: promotion now needs turns AND ≥3 sandbox
writes (`MIN_WRITES_FOR_SUGGESTION`, was `or ≥1`); writes are counted CUMULATIVELY across the session
(scratchpad `_session_sandbox_writes`) since a chat rarely writes 3 files in one turn; a big plan
(≥3 nodes) still qualifies alone. Title derives from the first NON-greeting user turn (`_GREETING_RE`
skips "hello"/"hey ghost"/"thanks", keeps "hi, can you build X"), falling back to first-non-empty.
**Needs prod restart.**

**Interface server audit (2026-07-12) — the web UI's live-log stream was DEAD in prod.** server.py's
default agent-log path was `/Users/vasilis/AI/Logs/…` (missing `Data/`) and the prod deployment
(`uvicorn server:app` via start-ghost-client.sh) never passes `--agent-log` (uvicorn owns argv) — so
`tail -F` followed a nonexistent file forever and the UI's face pulses / planner monologue never
fired. Fixed: correct default + `GHOST_AGENT_LOG` env override; log_streamer wrapped in a
restart-with-backoff loop (a dead tail used to end the stream until server restart); stream-cap hit
now CLOSES the upstream (was draining a discarded stream up to 30 min); `/api/stt` text-field-named
"file" → clean 400. Interface restarted + verified live on an emulated phone over TLS: 0 h-overflow,
SYSTEM ONLINE, 16 live log frames received. Mobile hardening: mic hold-to-talk gets
`touch-action:none`/callout suppression (JS already had the full touch lifecycle). SECURITY FLAG:
the launchd plist sets GHOST_API_KEY=**ghost-secret-123** — the exact guessable default the code
banned; the UI is TLS on 0.0.0.0:8080, so anyone on the LAN who guesses it gets full agent access →
operator should rotate it (bookmarks use `?key=`). **RESOLVED 2026-07-13** — key rotated to a
random secret in `~/Data/AI/.ghost_api_key`, exported by start-ghost-client.sh (overrides the
stale plist value), and agent auth ENABLED with the same key; see the 2026-07-13 (later 3) entry.

**Sandbox image baked to v4 (2026-07-12):** added `iproute2` (the `ss` port inspector) to apt and
`flask` + `python-chess` to pip in BOTH `sandbox/docker.py` (runtime provisioner) and
`sandbox/Dockerfile` (build-time), marker `.supercharged.v3`→`.v4`. "Host a web app / chess
service" requests were `pip install flask python-chess` mid-task (~24s serial thrash on the
critical path). The `ghost-agent-base:latest` cache was rebuilt to v4 already (incremental build on
the v3 base); a `v3` container re-provisions to v4 on next recreate. Sync guarded by
`test_provisioning_bakes_ss_flask_chess_and_stays_in_sync`. Takes effect on the next sandbox
container **recreate** (next prod restart).

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
- **#4 — UPDATE 2026-07-16 (B4 re-run, §6): dependency chain narrowed to ONE blocker — battery
  difficulty.** The 2026-07-09 chain was (a) battery too easy, (b) mediation ≈ 0, (c) dream starved.
  This re-run (with the day's memory fixes) RESOLVED (b): mediation went ~1% → ~71-100% (retrieval
  routing is fixed — the concrete win). With mediation healthy the outcome is a REAL null (treatment
  = control = 98%, McNemar p=1.0) but CEILING-CONFOUNDED — you can't detect improvement at a 98%
  baseline. So the idle-loop outcome question is now gated on ONLY (a): an expert-tier battery with
  baseline < 80%. (c) dream-starvation persists (auto_memories(seed)=0; lessons come from self_play +
  perfection_protocol, not dream). The single next step is the harder battery, NOT more repeats and
  NOT more retrieval work. Prior updates below for history.
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

- **[2026-07-14 post-July-hunt cohort review] open residuals** (the CONFIRMED-and-fixed items are
  in §6's 2026-07-14b entry). Still open:
  - **[concurrency] streaming final-generation tail escapes the semaphore + turn registry** —
    **RESOLVED 2026-07-15 (later).** The streaming path now defers `unregister` into a generator
    wrapper's finally (turn stays visible/cancellable for the whole drain) and the stream loop checks
    `is_cancelled` each chunk. Deliberately did NOT hold the permit across the drain (the sketch's
    suggestion): foreground-marking already protects the LLM slot, and holding it would couple turn
    serialization to client read speed. Tests: test_streaming_tail_cancellable.py. Docs: core/sessions.html.
  - **[host] `is_published_port` uses the CONFIGURED range, not the actually-published set** —
    **RESOLVED 2026-07-15 (later).** `DockerSandbox.published_service_ports()` records the set actually
    published at container (re)create (empty for a 2nd instance); `is_published_port(port,
    published_ports=…)` treats it as authoritative, None → configured-range fallback. Tests:
    test_sandbox_services.py. Docs: sandbox/services.html.
  - **[activity] `read_since` re-baselines to EOF on a shrunk ledger** (LOW, latent) — a truncation/
    rotation smaller than a saved watermark silently skips post-truncation records. No rotation code
    in-tree today. Fix: detect `size < offset` and re-read from 0.
  - **[games] tic-tac-toe `load` accepts turn/parity-impossible boards + double-winner boards** (LOW,
    SUSPECTED) — client owns state and it self-heals on the next `load`; reject a supplied `turn`
    that disagrees with mark parity.
  - **[self-play] `_invoke_template`'s broad `except TypeError` around the whole `fn(tier=)` call**
    (LOW, SUSPECTED) — a genuine TypeError from a template body silently falls back to `fn()` (wrong
    tier). Fix: detect kwarg support via `inspect.signature`, not a catch around execution.

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
  earliest `valid_from`). **WIRED LIVE 2026-07-14** — the dream cycle now calls it after
  `prune_stale_edges` via `Dreamer._compress_graph_nodes` (deterministic candidates from the new
  `propose_merge_candidates`; fuzzy pairs need a worker same-entity confirmation; capped 8/cycle;
  self-play ReadOnly wrapper still no-ops). Tests: test_graph_compression_temporal.py (7) +
  test_dream_graph_compression_wiring.py (13). Docs: memory/graph.html, core/dream.html.
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
- **[behavior] "project X" recall-routing variance** — **RESOLVED 2026-07-14.** Both suggested
  fixes landed: (a) a `manage_projects` `get` miss now runs `_not_found_with_recall` (vector
  `search_advanced`, MEDIUM-or-better band <1.15) and returns hits as a NON-error payload
  (`{"project": null, "memory_recall": [facts]}`) so the model answers without a strike or a second
  unprompted hop; (b) the `recall` tool description carries the mirror-image nudge ("ALSO the right
  tool when a question names a project manage_projects doesn't track"). Tests:
  test_iterative_recall_expand.py (12). Docs: tools/projects.html.
- **[search] Yandex fails over Tor** (low, known) — per-exit-node reachability; circuit rotation per
  attempt already implemented; search succeeds via other backends. MEASURE across exit nodes before
  changing the backend set (see `tor-search-reachability` memory). **Per-backend timeout cheap-win
  DONE 2026-07-15 (later):** `_engine_timeout` — mojeek keeps 18s, fast engines get 12s (grounded in
  the recorded 2026-07-08 latencies), freeing uncancellable race threads ~6s sooner on a blocked wave.
- **[response] long skill-list truncated** — **RESOLVED 2026-07-15 (later).** `manage_skills(action=
  'list')` now returns the COMPLETE compact inventory (acquired + composed) in one call with a footer
  steering the model to summarise built-ins by category, not reproduce every schema. Tests:
  test_acquired_skills.py. Docs: tools/acquired_skills.html.
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
- **[infra] torch leaked-semaphore at shutdown** — **RESOLVED 2026-07-15 (later).** Traced to TQDM
  (transformers' "Loading weights" bar → tqdm `get_lock()` creates a multiprocessing RLock / named
  posix semaphore, never reclaimed at SIGTERM; 441 occurrences in prod stderr). Fix:
  `tqdm.tqdm.set_lock(threading.RLock())` at `memory/vector.py` import (we never drive bars across
  processes). Verified 0 leaked (was 1). Tests: test_embedder_semaphore_leak.py. Docs: memory/vector.html.
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

### 2026-07-18 (later 5) — context-pressure governor: the xrick-session postmortem
Log eval of the Rick Dangerous feasibility session: ~60 whole-file reads (incl. dat_*.c
generated hex tables) → 398k est. tokens, 2 compactions, "successful" prune whose verbatim
last-6 tail still carried 5 parallel reads → 333k sent vs 262k n_ctx (HTTP 400) → recovery
reused the STREAMING payload on the non-stream API → 102KB SSE read as "non-JSON" → dead
turn; retry request ground 25+ min doing the same thing. Fixes:
- **Occupancy-aware ReadBudget** (agent.py dispatch): per-turn cap now min(old cap, bytes
  remaining below 80% of window given current conversation size). Zero capacity → every
  whole-file read refused with externalize-notes steer (tool_read_file: first-read exemption
  no longer applies at zero; ranged reads/search stay exempt).
- **Pressure steers** (handle_chat): prune actually fired → SYSTEM ALERT (write notes to
  disk, consult notes not sources, targeted evidence only); 2nd overflow same request →
  `_ctx_pressure_lockdown` (read budget pinned to 0) + synthesize-NOW steer.
- **`_cap_oversized_tail`**: post-prune enforcement — truncate largest non-system contents
  (head+tail kept) until ≤92% of max_context. Both _prune_context returns wrapped.
- **Recovery stream fix**: overflow recovery sets payload["stream"]=False; llm.py non-JSON
  retry also strips a leftover stream flag (SSE-body detection in the log line).
- **Generated-file sampling** (file_system): >96KB + (0x-dense head or avg line >240) →
  4KB SAMPLE ONLY + digest pointers, BEFORE the per-file cap.
- **Minors**: `command not found` fallback hint (file→od); browser navigate over Tor retries
  once with wait_until='commit' on timeout (Chromium can't SOCKS-auth for fresh circuits);
  work_log gains `commands` heads (execute-created state like git clones was invisible —
  caused the re-clone strike) + briefing renders "ran: …" when no files.
- Tests: tests/test_context_governor.py (new, 13); test_read_budget_overflow.py fixtures →
  multi-line (single-line char runs now correctly classify as data-shaped). Docs:
  core/agent.html, core/llm.html, tools/file_system.html, tools/browser.html.

### 2026-07-18 (later 4) — project-scope escape guards: CWD pin + off-project steer
Log eval of the recreated Prince-of-Persia project (requests f0fdb2f1/6f14407f): with the
project BOUND and constraints replayed, the agent cloned the repo and wrote
feasibility_report.md at the sandbox ROOT (/workspace/prince-persia-repo/), project dir
empty. Why: (1) the coding-turn CWD pin was STATIC text "SHELL CWD IS /workspace" even
though exec starts in /workspace/projects/<id> — the model obeyed the louder wrong signal
with `cd /workspace && git clone`; (2) absolute /workspace paths bypass scoping by design;
(3) the remap heal fires only on file-not-found — successful escapes hit no guard.
- **_render_cwd_pin(project_id)**: pin now names /workspace/projects/<id> when bound, with
  `cd /workspace && …` as the ✗ example; free-chat wording unchanged.
- **_offproject_target + dispatch steer**: successful file_system mutation to a
  root-absolute path, or execute with cd-to-root / root-path reference, gets ONE
  corrective steer per request (move files in, use relative paths). Cross-project
  absolute paths deliberately not flagged.
- Data repair moot: user deleted both Prince projects (tombstones 08:54/09:11) and
  cleaned the root strays by hand before the fix landed.
- Tests: tests/test_offproject_scope_guard.py (new, 10). Docs: core/agent.html.
- NOTE: agent restarted at ~09:11 — this fix (and anything after) needs the NEXT restart.

### 2026-07-18 (later 3) — recurring workspace tidy: the agent cleans up after itself
Operator complaint: projects leave screenshots/debug scripts behind for manual deletion.
Root cause: the DONE sweep fires ONCE on the transition; all verification/debugging
debris lands AFTER it (game project: 6 unswept screenshots between the 21:41 roll-up and
next morning). Fixes:
- **`tidy_project_workspace`** (core/workspace_cleanup.py): recurring, status-agnostic,
  much narrower than the sweep — deletes only categorical debris + unregistered media
  older than 24h that is NOT in the keep-set AND NOT referenced by any source file
  (basename scan ≤512KB/file — a sprite sheet index.html points at is an asset, not a
  screenshot). Source files never deleted regardless of registration. One
  `workspace_tidy` event per pass that removed something.
- **Watchdog phase 2.7d**: idle, 6h cooldown, walks all project workspaces (24h age
  gate). Sandbox-ROOT strays (repo clones, analysis/, chess/) deliberately untouched —
  that's the free-chat workspace.
- **`manage_projects action=cleanup`**: explicit user-triggered tidy, NO age gate.
- Note: re-DONE after the defect-reopen flow re-fires the full sweep (hook fires on
  every transition) — verified in code, no change needed.
- Tests: tests/test_workspace_tidy.py (new, 12). Docs: core/workspace_cleanup.html,
  tools/projects.html.

### 2026-07-18 (later 2) — write-only project plumbing gains read sides
Follow-up to the work-log session: the five records catalogued as persisted-but-invisible
are now all readable:
- **Deliverables manifest**: `ProjectStore.list_deliverables` (deduped kind='file'
  artifacts) → briefing DELIVERABLES section (≤12 paths, "+N more") + status snapshot key.
- **tool_call/note/url artifact payloads**: new `manage_projects action=artifact_list`
  (project-wide / task_id scope / artifact_kind filter / limit, non-file payloads
  truncated 400 chars) — the artifact store previously had a write action and NO read.
- **Retrospectives**: `generate_retrospective` had no caller; now rendered lazily for
  terminal-status projects (briefing RETROSPECTIVE: summary + ≤3 what_failed + measured
  effort) and in the status snapshot.
- **dream_digest events**: LAST DREAM DIGEST one-liner in the briefing (newest event).
- **Cost columns**: `actual_cost` finally WRITTEN (advancer stamps tick seconds on all
  three finalize paths, incl. failed builds) and READ (retro `total_actual_cost_s`).
- Tests: tests/test_project_readside_plumbing.py (new, 13). Docs: core/prompts.html,
  core/project_advancer.html, memory/projects.html, tools/projects.html.

### 2026-07-18 (later) — project context closed-loop: work log + defect reopening + briefing gaps
Root cause of "agent forgets project work already done / pending": interactive turns NEVER
wrote to the project store (agent.py had zero store writes — everything relied on the model
volunteering task_update, which has no trigger once tasks are closed). Case study: game
project bd75420e2d96 rolled DONE at 21:41; 6 evening debugging requests + the 06:20
root-cause fix left ZERO store events; briefing kept saying "DONE, no open tasks"; the
morning turn re-read all files and re-derived the bug from scratch.
- **Work log (write side)**: memory/projects.py `add_work_log`/`recent_work_logs` — one
  bounded `work_log` event per working request (request head, ≤12 files, tool counts,
  verifier-aware outcome, note head). agent.py: accumulators reset at request start
  (after conversation reconcile), filled in the dispatch results loop (project-scoped
  successful mutations + execute/browser/vision), written in `_finalize_and_return`
  before trajectory recording. No LLM cooperation required.
- **Work log (read side)**: build_project_briefing gains RECENT WORK LOG (5 newest,
  "trust this before re-reading files") — rendered for DONE projects too, which is
  exactly when it's the only record. `action=status` snapshot gains `recent_work_log`.
- **STUCK TASKS in briefing**: FAILED/BLOCKED tasks + failure_reason were write-only
  (OPEN filter excludes both; DONE SO FAR only shows DONE) — now surfaced (≤4).
- **Defect reopening**: bug-report intent (existing repro-first gate) against a DONE
  project → `_note_defect_on_done_project` adds "FIX (defect): <head>" task; add_task's
  DONE→ACTIVE semantic reopens; deduped while an open defect task exists.
- Explorer's full-subsystem map also catalogued remaining write-only plumbing (artifacts/
  deliverables manifest, tool_call artifact payloads, retrospectives, dream_digest events,
  costs) — deliberately NOT surfaced yet; work log covers the acute gap.
- Tests: tests/test_project_work_log.py (new, 17). Docs: core/prompts.html,
  core/agent.html, memory/projects.html, tools/projects.html.

### 2026-07-18 — overnight log-eval fixes: fix-verify unblocked + self-play infra fairness + idle no-op skips
From evaluating the 2026-07-17 21:01 → 07-18 06:25 log (game project bd75420e2d96 + idle loop):
- **World-changed reset** (core/strikes.py + dispatch pipeline): a successful file_system
  mutation clears the no-progress observation ledger + steer set + batch trip. Every fix-verify
  turn (26/3B/72/1E/91) had its post-fix navigate killed by "repeated 2x with no new info";
  in 3B the verifier then REFUTED for missing post-fix evidence — two guards fighting.
  `execute` probes deliberately do NOT reset (probe loops must still trip).
- **Batch dedup of identical read-only calls** (core/agent.py dispatch): the 22:14 batch ran
  144 byte-identical file_system reads; dups now execute once, result fanned out, breaker
  counts unchanged. Mutating dups never collapse.
- **Self-play validator crashes = INFRA_ABORT** (core/dream.py): score-time validator crash no
  longer records FAILURE/Δ=-1.0/score/adversarial-fingerprint (04:50 run: solution.py exit 0,
  broken validator charged the agent). Pre-flight dry-run now also fails on module-scope
  AttributeError; new `_datetime_misuse` AST lint rejects both observed datetime import-style
  crashes at generation time (+ prompt rule 13).
- **Challenge diversity guard** (memory/frontier.py + dream.py): rolling 12-head window of
  recent LLM challenges; containment + shared-mock-filename bonus ≥0.60 rejects reworded
  repeats (4/6 overnight were the same transaction_log.csv fraud scan); recent heads also fed
  forward as negative examples in the gen prompt.
- **Conversational-trigger gate** (memory/lesson_quality.py): mistake-bearing lessons no longer
  bypass quality — triggers that are raw chat ("proceed with the next task", "it still does
  the same… notify me in slack") reject; user-question triggers ("How do I parse JSON?") kept.
- **Skip-if-unchanged idle gates** (distill/collector.py `corpus_fingerprint` + agent.py): PRM
  retrain, router retrain, and reflection tick skip when the trajectory corpus is byte-identical
  to the last completed pass (overnight: 3 identical refits each; 8× "reflected 0/60" walks).
- **Game status**: the 06:20 menu-bounce fix VERIFIED working headless (overlays clear, 60fps
  loop, no console errors) — but the canvas still renders BLACK: `Animation "undefined" not
  found` every frame (animation.js startAnimation called with undefined name). That bug is
  still open in the game project; the agent never got a clean in-turn verification of it
  (loop-breaker conflict above — now fixed).
- Tests: test_strike_ledger.py +3, test_dispatch_pipeline_extraction.py +4,
  test_dream_synthetic.py +1, test_validate_challenge_quality.py +9,
  test_frontier_diversity_guard.py (new, 7), test_distill_collector.py +4,
  test_lesson_quality_gate.py +1 class; 2 old fixtures renamed (conversational-trigger
  collisions). Docs: core/strikes.html, core/dream.html, core/agent.html, memory/frontier.html,
  memory/skills.html.

### 2026-07-17 (later 8) — A3 trace fixes: remap note on failures + syntax-reject would-be snippet
From evaluating the Prince-of-Persia parser request (A3, 22 turns, late-REFUTED correctly):
- **Remap teaching note rides failed runs** (tools/execute.py): the "/workspace → project scope"
  heal taught the model the bare-relative-path rule only on exit-0 runs; A3's remapped runs
  failed for their own reasons → zero learning across 5 heals/22 turns. Note now rides every
  adopted remap, with "(failed for reasons UNRELATED to the path)" on non-zero exits.
- **Syntax rejection shows the rejected lines** (tools/file_system.py): the tool result already
  carried "msg (line N, col M)" but the LOG line didn't (operator opacity), and the model still
  couldn't SEE its mistake — it blamed "hidden characters" and corrupted the file via an
  unguarded patch script. Log line now carries the detail; the rejection includes a numbered
  `_would_be_snippet` of the rejected content around the error line ('>' marker).
- Tests: test_execute_path_and_exit1_heal.py +1, test_replace_marker_leak_guard.py +2. Docs:
  tools/execute.html, tools/file_system.html.

### 2026-07-17 (later 7) — CLI inline images
`interface/externals/cli/ghost` now draws reply-referenced images (`![…](name.png)`) in the
terminal, fetched from the sandbox via the existing `/api/download` (in-memory, 25MB cap, ≤4 per
reply; http/data schemes skipped). Rendering AFTER the reply settles — the escape protocols are
raw byte streams a rich Live repaint would shred. Auto-detection: iTerm2/WezTerm OSC-1337 →
kitty graphics (PNG-only `f=100`; non-PNG transcodes via Pillow) → universal half-block `▀`
truecolor fallback → none; `GHOST_CLI_IMAGES=off|iterm|kitty|halfblock` overrides. `/download`
of an image renders it too. Pillow added to the PEP-723 deps, OPTIONAL at runtime. Verified
end-to-end against the live agent (upload → reference → fetch → half-block draw). Tests:
test_ghost_cli.py 11→22 (ambient LC_TERMINAL leaks into detection tests — scrub it). Docs:
interfaces/cli.html. No agent restart needed (client-side; bin symlink serves the live copy).
FIRST LIVE USE found two gaps, fixed same hour: (1) the model embedded the FULL API path
(`![…](/api/download/gen_x.png)`) → double-prefixed fetch → 404 — refs now normalized
(api/workspace/sandbox prefixes stripped, anchored; 404 on a pathed ref retries the flat
basename); (2) operator runs tmux-on-iTerm2 — tmux swallows OSC-1337 unless allow-passthrough
(default off ≥3.3), so auto-detection under tmux now picks half-block (always visible);
explicit GHOST_CLI_IMAGES=iterm wraps in DCS passthrough framing (needs `set -g
allow-passthrough on`). Tests 22→28 (incl. TMUX ambient scrub in detection tests).
SECOND live report: passthrough images VANISH on tmux resize/pane-switch — inherent (overlay;
tmux repaints from its char grid). Fix: pure-python SIXEL encoder (`_render_sixel`, 256-color
quantize + RLE, ~0.4s/640px) — a sixel-built tmux (operator: Homebrew 3.7b) consumes and OWNS
the image, repainting it across redraws; emitted RAW, never passthrough-wrapped. Auto under
tmux: sixel when tmux≥3.4 + iTerm2/WezTerm + Pillow, else half-block; GHOST_CLI_IMAGES=sixel
forces. Tests 28→32.

### 2026-07-17 (later 6) — INCIDENT: "worker broke, cache doesn't work, tools not firing" — triage + fixes
Operator report after the later-5 deploy. Three distinct causes, all resolved:
- **Cache + tools** — self-inflicted by `--no-native-tools`: schemas moved into the prompt but the
  boot warmup still prefilled the NATIVE-mode head (~5.7K tok vs the new ~84K-char request
  prefix) → every conversation re-paid a ~20K-token prefill, turn-1 latency 12s→47s; separately
  the GBNF trigger armed INSIDE thinking (model drafts literal `<tool_call>` while reasoning) and
  hard-killed tool turns. REVERTED same hour (native ON, grammar opt-in-off). Re-attempt
  prerequisites documented in the launcher comment block.
- **Worker (Nova) — NOT my deploy, NOT Nova: macOS Tahoe Local Network privacy.** The agent runs
  as a SYSTEM launchd daemon (UserName key); Tahoe silently denies daemons access to physical-LAN
  addresses. PROOF: (a) tcpdump — the agent's SYNs to 192.168.0.20:8088 never reached the wire
  while a terminal curl handshook cleanly in the same capture; (b) a one-shot probe daemon
  bootstrapped into the system domain got `errno 65 No route to host` on the LAN IP but **200 on
  the tailnet IP** and loopback. Loopback + internet unaffected — which is why only worker/image
  nodes died. Onset ~17:00-17:30 with no reboot/update — Tahoe's policy attribution is opaque;
  mechanism proven even if the trigger moment isn't.
- **FIX:** `--worker-nodes http://100.83.184.117:8088|Nova` and `--image-gen-nodes
  http://100.122.46.101:8000|Ghost` (tailnet IPs; `compute_tor_proxy`'s CGNAT rule already
  bypasses Tor for 100.64/10). LIVE-VERIFIED: verify → Nova → CONFIRMED (100%) in 4.3s.
  Alternative left to operator: grant Local Network to the venv python in System Settings and
  revert to LAN IPs. Diagnostic ladder that worked: fresh-process repro (isolate code) →
  `ps eww` env diff → tcpdump (in-host vs wire) → same-domain probe daemon (context isolation).

### 2026-07-17 (later 5) — THREE features: GBNF tool grammar, LLM record/replay, counterfactual phase 1
Operator-approved evaluation → "proceed with all 3". All landed with kill switches.
- **Grammar-constrained tool calls** (core/tool_grammar.py + payload wiring + launcher
  `--no-native-tools`): lazy GBNF from the registry schemas, PATTERN trigger on `\n<tool_call>`
  (newline-anchored — a quoted tag in thinking can't arm it). LIVE-VALIDATED before wiring:
  the running llama-server accepts per-request grammar+lazy triggers (word-type needs preserved
  tokens → pattern-type used); full 39-tool grammar (26KB) compiled and emitted canonical calls;
  `action='bogus_action'` coerced to a legal enum. GOTCHA FOUND ON PROBE: whitespace padding in
  value rules let the sampler stall on tabs instead of committing — value rules are TIGHT.
  Native upstream parsing RETIRED (its two corruption shapes hit 3× in two weeks on VALID
  output); the agent's XML parser consumes the output. XML schema block returns to the prompt
  (~7K tok, prefill-cached). **GRAMMAR DEMOTED TO OPT-IN same session** (req 9f1c3173): first
  production request hard-failed — with THINKING on, the model drafts literal `<tool_call>`
  inside reasoning, the pattern trigger armed mid-think, generation died at `<tool` every
  retry. The probes' blind spot: all ran /no_think. Default now OFF (`GHOST_TOOL_GRAMMAR=1`
  re-enables); next step is llama.cpp PATTERN_FULL think-aware triggers, validated on a
  thinking turn. `--no-native-tools` STAYS — XML-parser path live-verified post-restart
  ("list your projects" → clean manage_projects call, no repairs/stalls).
- **LLM-boundary record/replay** (core/llm_recording.py): `GHOST_LLM_RECORD=1` records every
  chat_completion + route() call (payload/response/ordinal/request-id) to
  system/llm_recordings/; OFF by default — raw prompts bypass redaction (operator-only
  retention). `ReplayLLMClient` = deterministic stub-replay for fixture minting. Byte-exact
  LIVE re-generation rejected as a goal (Metal + prefix-cache ≠ byte-stable).
- **Counterfactual replay phase 1** (core/counterfactual.py): concluded self-play challenges
  persist (prompt+setup+validator+outcome); ~1 idle self-play slot in 4 replays a past
  challenge via new `synthetic_self_play(injected_challenge=…)` seam (generation skipped,
  journal path guarded with `not gen_ok`); FAILURE→SUCCESS = "generalized",
  SUCCESS→FAILURE = "regression" → hydrated lessons QUARANTINED (skills.py:
  `quarantine_lesson` + `_filter_quarantined` chokepoint over both retrieval surfaces; kept on
  disk, never deleted) + notify-severity ledger record. Chat trajectories stamp
  `extra.hydrated_lessons` (attribution substrate). Scope: validator-backed tasks only;
  user-turn replays need workspace snapshots (phase 2).
- Tests: test_tool_grammar.py (13), test_llm_recording.py (9), test_counterfactual.py (12).
  Docs: core/agent.html, core/llm.html, core/dream.html, memory/skills.html. Deploy note:
  agent restart picks up grammar+native-off together (launcher edit).

### 2026-07-17 (later 4) — log-eval fixes: interaction cap, click bound, name-case, dup guard, re-anchor
Operator asked for a model-vs-harness evaluation of the day's 4 requests (WebOS drag session,
reqs 75/AF/43), then "proceed with all". Five harness fixes:
- **Verifier interaction-claim cap** (the big one): AF's drag fix got CONFIRMED (100%) from text
  entailment + a load-clean WEB-EXEC probe — and was still broken (req 43 = the user reporting
  it). New `_is_interaction_intent` + `_has_interaction_evidence` (reads `STATUS: OK`/`OP:` from
  browser results); pointer-behavior claims with no successful click/interact this turn cap
  CONFIRMED at 0.6 (below all ≥0.7 consumption gates), reasoning annotated. Same philosophy as
  the req-70 web-exec-inconclusive cap.
- **Browser click bounded** (runner): attached ≠ actionable — a hidden start-menu item passed the
  2026-07-14 probe then page.click burned Playwright's full 30s default, raw TimeoutError, no
  steer (so the model abandoned behavioral testing). Click now runs with timeout=probe_ms (≤8s);
  timeout re-raises with the hidden-until-opened explanation + the op='interact' escape.
- **Service names case-insensitive** (`_resolve_name` in sandbox/services.py): restart 'WebOS' vs
  registered 'webos' missed → duplicate service, port conflict, kill dance. stop/restart/status/
  logs/start now resolve exact-first then unique case-insensitive; registered spelling wins.
  (The workspace still carries the leftover WebOS.cmd.sh twin from the incident — harmless, the
  running entry is 'WebOS'.)
- **file_search filename==pattern guard** (sibling of the replace content==replace_with guard,
  2026-07-05): upstream value-duplication put the search pattern in 'filename' → rg on a
  nonexistent path. Now heals (drops corrupt filename, searches workspace, NOTE in result).
  Plus: `_repair_native_tool_calls` firing now logs raw pre-repair calls (4 KB) — was
  undiagnosable from traces.
- **AUTO-DIAGNOSTIC re-anchor**: the failure-context+listing flood made the model re-run the
  PREVIOUS request's flow for 3 turns (43/22-24); the injection now ends with "REMINDER — the
  CURRENT user request…".
- Deferred with reasons: prefix-growth turn-1 latency (design tradeoff, not a bug); surfacing
  fresh post-mortem lessons into in-flight fixes (real feature, own design pass — noted: AF's
  auto-lesson described exactly the generalization whose absence caused 43).
- Tests: test_log_eval_fixes_20260717.py (10), test_sandbox_services.py +4, 
  test_file_system_search_container_path.py +3. Docs: core/verifier.html, core/agent.html,
  tools/browser.html, tools/file_system.html, sandbox/services.html.

### 2026-07-17 (later 3) — ghost CLI moved into the repo (interface/externals/cli/)
The terminal client lived ONLY at `~/Data/AI/bin/ghost` — outside version control, invisible to
the test suite, and un-diffable against anything (the exact "device copy accumulates live fixes"
trap the uConsole hit on 2026-07-13). Moved to `interface/externals/cli/ghost` (exec bit kept,
PEP-723 inline deps unchanged); `~/Data/AI/bin/ghost` is now a symlink to the repo copy, so the
`ghost` command (that dir is on PATH) and its `uv run` shebang behavior are unchanged. Verified
live: `ghost --health` against the running agent through the symlink. Tests:
tests/test_ghost_cli.py (11 — loads the extensionless script by path; location + symlink
contract, formatting helpers, error_of shapes, base-URL normalization, key precedence). Docs:
docs/interfaces/cli.html (new) + sidebar links on the 5 sibling interface pages and index.html.

### 2026-07-17 (later 2) — multi-turn replies smoothed: narration + double-summary scrub at finalize
Operator: replies on multi-tool fixes read as the raw loop transcript (WebOS minesweeper turn:
"Let me fix both:", "Now add the resize logic:", summary stated twice around the verify/restart
step) — "how can we fix the replies to be smoother?". Option picked (of three): finalize scrub +
prompt rule; no client changes (live stream keeps narration as progress; the delivered/persisted
reply is cleaned).
- **New `core/reply_smoothing.smooth_reply()`** — pure text→text, two removable shapes only:
  (1) non-final connective narration paragraphs (≤300 chars, "Let me/Now/Good,/I'll/…" starters;
  lists+fences exempt); (2) superseded summary groups — ':'-lead-in + list restated by a later
  near-duplicate group (word-Jaccard ≥0.55, size ≥0.6) — last statement wins. Fence-atomic
  splitting, final block never dropped, fail-open, idempotent.
- **Wired in `_finalize_and_return`** after the adjacent-duplicate collapse, BEFORE the verifier
  gate (verdict judges delivered text). Gated on ≥2 non-synthetic tool runs — conversational and
  single-tool replies never rewritten. Logs "Reply Smoothing — trimmed … chars" when it fires.
- **Prompt sharpened** (EXECUTION MODE): no working notes alongside tool calls, no summary until
  the whole task (incl. verification/restart) is done, never restate a summary. NOTE: changes the
  byte-stable system prefix → one-time main-node cache re-prefill on next boot.
- Tests: tests/test_reply_smoothing.py (12; verbatim WebOS transcript is the fixture — asserts all
  6 narration paragraphs dropped, pre-verify summary superseded, final summary + port line intact,
  diagnosis kept conservatively). Docs: core/agent.html new section.

### 2026-07-17 (later) — activity banner tamed: notify-only in chat + on-demand `introspect activity`
Operator: the "Background activity while you were away" block (dream ×2, PRM/router/calibration
refits…) "is not very elegant — only if asked?". Decision (operator-picked from three options):
notable-only banner + on-demand report.
- **Banner** (`render_activity_digest` + the finalize call site): renders `severity="notify"` only —
  scheduled-task conclusions, deliberate `notify_operator` messages, failures. Info-severity
  maintenance no longer auto-surfaces in chat (it's already on the live pretty-stream; Slack push
  was always notify-only). Watermark still advances over info records (seen-but-silent). Identical
  (phase, summary) repeats collapse to one `(×N)` line.
- **On-demand**: new `introspect action='activity'` → `render_activity_report` — ALL severities,
  newest first, timestamped ages, ×N-collapsed, `hours` window (default 24h, cap 14d) + `limit`
  (30/100). Reads the ledger from byte 0 (≈100KB/week), does NOT consume the banner watermark.
  Branches before the tool's self_model gate (works with selfhood disabled); registry now passes
  `context` into tool_introspect; tool description names "what did you do while I was away?" so the
  semantic router finds it. Kept OUT of `_BOOKKEEPING_TOOL_NAMES` deliberately — the report is real
  evidence for a reply summarizing it.
- Tests: test_autonomous_activity.py 48→57 (severity gate, dedupe, report render, wiring pin on
  `severities=(_SEV_NOTIFY,)`), test_selfhood_introspect_tool.py 17→22. Docs:
  core/autonomous_activity.html (consumer 1 updated, consumer 4 added).

### 2026-07-17 — verifier evidence STARVATION: slack redistribution + URL squeeze + repair-leak guard
Operator asked for an evaluation of the latest request (req 4dab5067, "whats the news" →
naftemporiki RSS skill + weather). The answer was fully supported by the feed, yet the verifier
REFUTED it twice (sync → repair round, then LATE 100% → lessons scrubbed, corpus backfilled
`failed`, bogus next-turn correction queued). Root cause was NOT the worker model or Greek text
(first hypothesis, discarded on evidence): the 2026-07-16 one-pass newest-heavy budget split let a
106-char weather report — the NEWEST tool — hold 65% of the 4000-char evidence budget unused while
the 4KB headlines feed was cut mid-item #4. Every refuted item (#5–#10) sat past the cut; the
verifier was right about the sliver it was shown.
- **FIX 1 — `_collect_verifier_evidence` two-pass allocation.** Weighted caps as before, then
  unused slack is redistributed to still-truncated items (newest first). Plus
  `_squeeze_evidence_noise`: tracking-length URLs (`https?://\S{72,}` → 64-char stub + `…`) are
  trimmed before budgeting — zero entailment value, ~70% of the RSS payload. Replayed on the
  failing turn's real trajectory data: all 10 headlines + weather now fit in 2212/4000 chars.
- **FIX 2 — repair directive standalone-rewrite suffix.** The repaired reply had leaked
  checker-facing dialogue to the user ("You're right — I was interpreting and embellishing…" —
  addressed to a verifier the user can't see, about a draft the user never received). Both repair
  directives now append `_REPAIR_STANDALONE_SUFFIX` (agent.py): no acknowledgements, no apologies,
  no mention of verifier/review/correction.
- **FIX 3 — `duration_s` stamped on chat trajectories** from the pretty-log request clock (new
  public `request_elapsed_s()` in utils/logging). Was schema-default 0.0 on every turn.
  `tokens_in/out` stay 0 — needs usage accumulation on the streaming client path; scoped out.
- **Corpus repair:** appended an `operator_overlay` line to trajectories/corrections.jsonl
  reverting 2466f380 `failed` → `unknown` (overlay reader is last-write-wins; original line kept
  for audit). The in-memory bogus pending-correction cleared with the deploy restart.
- Residual noted, not built: translation slips in the reply itself (ΟΠΕΚΕΠΕ→"OPEKED",
  "ξενόδουλη"→"tourist opposition") — genuine errors the verifier didn't catch; cross-lingual
  claim-vs-evidence quality on the worker model remains unmeasured.
- Tests: test_verifier_evidence_window.py +6 (2026-07-17 section), test_wiring_trajectory_logging.py
  +2. Docs: core/verifier.html (three new 2026-07-17 subsections).

### 2026-07-16 (later 3) — verifier triage: verify-sized timeout + widened evidence window + finalize stagger
Operator report: "verifier either refutes valid answers or timeouts on work node." Log forensics
(requests 73/35/E0/BD, the Athens food-research turns) found two distinct mechanisms that compounded:
- **Timeouts.** With `--critic-nodes` absent from the live flags, every verify rode `route()`'s 12s
  `_ROUTE_TIMEOUT_S` (sized for sub-second routing chores) to the single worker node. A verdict takes
  7–11s UNCONTENDED (measured in-log); at finalize, verify + hydration-judge fired the same second and
  the loser blew the ceiling — `Nova: ReadTimeout` on 4/17 verify calls that day. Worst case (req 35):
  the gate verify died → the "Everest pizza / world champion June 2026" hallucination shipped
  unchecked; only the LATE async verdict caught it (correctly) a turn later. FIX 1: `route()` accepts a
  per-call `timeout`; verifier passes `_VERIFY_WORKER_TIMEOUT_S` (45s, `GHOST_VERIFY_WORKER_TIMEOUT`).
  FIX 3: `_attach_late_verdict_handler` publishes `_deferred_verdict_task`; `_judge_hydration_safe`
  defers (bounded 90s, `_HYDRATION_JUDGE_STAGGER_S`) to the in-flight verdict — back-to-back, not
  colliding.
- **False refutes.** The gate judged the WHOLE answer against only the LAST substantive tool output
  (4000 chars). Req 73: answer built from two loaded sources, last fetch was a 403 → REFUTED "no data
  was retrieved" → wasted repair round. FIX 2 (careful): `_collect_verifier_evidence` — last 3
  substantive outputs, chronological, `[tool_name]`-labelled, newest-weighted 50/30/20 budgets, total
  ≤4000 incl. labels (single-tool turn = old behavior); claim-shaped paths only (code-shape still
  audits the one run); prompt now says one failed tool doesn't refute parts supported by others AND
  specifics in NO output are fabrications (keeps the req-35 catch). `last_tool` semantics untouched.
- NOTE: req 35's LATE REFUTED was CORRECT — the verifier's judgment is fine when it fires; it was
  evidence-starved and time-starved. Residual (not built): loop-exit repair `_vtask` that overruns its
  25s budget is abandoned and the post-loop gate spawns a fresh verify — duplicate Nova compute;
  adopting the in-flight task is unsafe when repair mutated the answer, needs care. Also consider
  re-enabling `--critic-nodes` (120s budget, off-host) as the verifier's proper home.
- Tests: test_verify_worker_timeout.py (6), test_verifier_evidence_window.py (13),
  test_finalize_stagger.py (7); route-timeout source pin in test_worker_thinking_and_prompt_clarity.py
  updated. Suite **7854 passed**. Docs: core/verifier.html, core/llm.html, core/agent.html,
  core/bus.html. NOT yet deployed (operator relaunch decision pending).

### 2026-07-16 (later 2) — two features: grounded file-artifact verification + reactive watch scheduler
Two operator-approved features, both grounded in measured need.
- **Grounded file-artifact verification (`core/agent.py`, `core/verifier.py`).** The agent's #1
  most-retrieved real lesson (ret=55) is "prematurely declared task completion … without showing the
  actual content" — it claims a file deliverable that's missing/empty; B4 independently showed the LLM
  verifier CONFIRMs at 100% while the agent errs. This is the general form of the existing web-exec
  "execute, don't trust" override: `_claimed_deliverable_files` extracts filenames the answer presents
  as PRODUCED (anchored on completion verbs saved/wrote/created/… — ignores files merely READ and
  present-tense "the script writes to X"), `_verify_file_artifacts` re-reads each under the
  project-scoped sandbox HOST path (direct FS, no docker exec), and a missing/empty claimed deliverable
  → REFUTED(0.9) overriding a plausible text CONFIRM and feeding the same bounded auto-repair loop.
  Low false-positive by construction (an input file the agent read still exists → passes; only a
  claimed-but-absent deliverable refutes). Runs last in `_compute_verifier_verdict`. Tests:
  test_grounded_file_verify.py (17).
- **Reactive condition-watching scheduler (`tools/tasks.py`, `main.py`, `registry.py`).** The scheduler
  could only fire on a clock (cron/interval). New `manage_tasks(action='watch')`: polls a shell
  CONDITION every interval_secs in the sandbox and fires its reaction prompt only on the transition to
  true (edge-triggered — exit 0 = condition true, shell `if` semantics), with the check output injected.
  Reaches LAN/tailnet directly (sandbox egress guard) for real ops checks: `grep -q ' ERROR ' log`,
  `! curl -sf https://host/health`, `[ $(metric) -gt 90 ]`. Edge state (`last_fired`) persists so a
  restart doesn't re-fire an already-true condition; `_run_watch_condition` bound in main.py alongside
  the proactive runner, restore path handles watch records. LIVE-VERIFIED: registered a watch via a
  chat turn → it polled → `condition became TRUE — reacting` → reaction dispatched with context → (the
  agent even recognized the test and stopped the watch itself). Tests: test_watch_condition.py (8).
- Both deployed (plain-kill → pid 53638, health ok). Suite **7828 passed**. Docs: core/verifier.html,
  tools/tasks.html. NOT built (deferred): the numeric self-check half of grounded verification (re-derive
  a computed answer) — harder to ground generically; the artifact check covers the top lesson.

### 2026-07-16 (later) — lesson-quality gate: the playbook was 28% non-actionable noise
Grounded the "next improvement" in the agent's OWN data (post-mortem queue is off in prod, so used the
live skills_playbook.json). Finding: the playbook's top-retrieved "lessons" were dream/self-play
OBSERVATIONS, not mistake-and-fixes — `mistake="none"` pseudo-lessons like "When playing live chess
against Vasilis, provide continuous coaching commentary" (84 retrievals, the single most-retrieved
item), "On a regex_parse task that has a familiar shape…", "The user prefers ripgrep". Measured: 23/50
lessons mistake-less, drawing **28% of ALL playbook retrievals** — noise injected on every relevant
turn, diluting the retrieval-routing win from earlier today.
- **Gate at the write chokepoint.** The dream heuristics loop had an actionability gate since
  2026-07-13, but it lived INSIDE dream so self-play/other producers bypassed it. Moved
  `_is_actionable_heuristic` + constants to a leaf module `memory/lesson_quality.py` (shared with
  core.dream, no import cycle) and added `is_actionable_lesson(mistake, solution, task)` at
  `SkillMemory.learn_lesson` — the single chokepoint covering EVERY producer. Logic: a real-mistake
  lesson is a genuine correction (always pass); a mistake-less entry must have an actionable solution.
  `verified` gets NO bypass (it would only exempt a verified observation). Caught + fixed my own bug
  pre-ship: dream stores `task=solution[:80]`, so a `solution==task` degeneracy check would have
  wrongly dropped short valid rules — removed it (the actionability check alone catches the 11 real
  drops).
- **Retroactive prune** (fcntl-safe against live prod via `remove_by_trigger(memory_system=None)`,
  which load-modify-saves under the same lock prod uses): 50 → 39 lessons, all 11 non-actionable
  removed, 0 remaining. Backup: `skills_playbook.json.pre-quality-prune-20260716`.
- **Orphaned vector twins cleaned in-process.** A second PersistentClient on the live Chroma dir risks
  HNSW corruption (vector.py documents this), so the twin delete had to run INSIDE prod: added
  `VectorMemory.delete_skill_twins(triggers)` (locked, precise `type=skill`+`trigger` key, returns
  before/after count) + `POST /api/memory/delete_skill_twin` (companion to /correct, /delete) + the
  readonly façade's mutator list (the guard test caught the new writer — working as intended). Deployed,
  POSTed the 11 triggers → removed 11 (7315→7304 docs), verified idempotent (re-POST removed 0), prod
  healthy (no HNSW corruption). Tests: test_delete_skill_twins.py (3).
- This is the CONTENT-quality complement to the morning's retrieval-ROUTING fix: routing now surfaces
  lessons (mediation ~1%→~85%), and this stops ~a quarter of what surfaces from being noise. Deployed
  (plain-kill → pid 50686; prod reloaded the pruned 39-lesson playbook). Tests:
  test_lesson_quality_gate.py (19). Suite 7800 passed. Docs: memory/skills.html, core/dream.html.

### 2026-07-16 — B4 re-run: retrieval routing FIXED (mediation ~1%→~85%); outcome still ceiling-confounded
Ran the B4 grounded outcome battery overnight with the day's memory-retrieval fixes in place, to re-ask
"do the idle-learning loops improve outcomes." Operational notes: the harness must be launched FULLY
DETACHED (`os.setsid` double-fork — macOS has no `setsid` binary) because `run_in_background` tasks are
reaped by the session at ~2h (killed the pilot at 91/105). Host is memory-marginal (36GB, llama-server
18.4GB resident); the concurrency probe tasks (conc_*) spawn worker pools that tipped it into swap
thrash + a stuck turn (operator spotted "server idle"). Root causes: a pilot-orphan sandbox container
never cleaned up (the reap gotcha) + the conc_* memory spikes. Fix: cleaned containers, excluded the 3
conc_* tasks (`b4_battery_noconc.json`, 32 tasks), re-ran 3×3 detached → completed clean in 9h30m
(swap dropped 2.7GB→1.08GB after cleanup). Report: `ablation_out/b4-20260715-trim/`.
- **HEADLINE WIN — retrieval routing is FIXED (the prior B4's fatal flaw).** Prior B4: mediation ≈ 0
  (lessons surfaced in 1/96 probes), which made every outcome number uninterpretable. This run:
  mediation control 100%, treatment 86%, uniform 71% (probes where a playbook lesson's retrieval
  counter bumped). The morning's memory fixes (RRF anchoring, session self-hit exclusion, vector
  match gate, + the 2026-07-09 domain-rescue) demonstrably closed the loop. This is the concrete,
  validated payoff of the session.
- **Outcome NULL, but ceiling-confounded.** control 94/96, treatment 94/96, uniform 95/96 (98-99%);
  treatment-vs-control McNemar p=1.0, task-stratified mean delta +0.000 p=1.0. The 5 failures are
  scattered near-misses (off-by-a-few compute/parse errors) with no arm pattern — noise. Per the
  pre-registered reading (outcomes-null + mediation healthy → "idle output doesn't transfer at this
  scale") the null is now REAL (not an instrument artifact) — BUT you can't detect improvement when
  the baseline passes 98%. Same battery-difficulty wall as B3/prior-B4. The idle-loop outcome question
  is now cleanly gated on BATTERY DIFFICULTY, not retrieval → the real next step is an expert-tier
  battery (baseline <80%), the #4 item.
- **#27b frontier — still a WASH → uniform stays default.** Frontier ties uniform on self-play yield
  (1=1 in all 3 repeats) AND weak-cluster pass (47/48=47/48). No evidence frontier out-yields uniform;
  consistent with the 2026-07-09 flip. No change: frontier opt-in, PRM stays. (Caveat: excluding conc_*
  dropped the concurrency weak-cluster from the #27b delta; everything tied on the other 3 clusters.)
- **Dream STILL starved.** auto_memories(seed)=0 everywhere → the entropy gate had no material; the
  lessons came from self_play + perfection_protocol, not dream. Dream-specific value remains inert
  (needs a trajectory-shaped seed source — known open item).
- New tool: `scripts/ablation_monitor.py` (progress/ETA monitor for a running trackb4 run; counts
  DRIVER-tagged turns so self-play isn't miscounted). Tests: test_ablation_monitor.py (11).

### 2026-07-15 (later) — §4 residual burn-down: relevance gate, streaming-tail cancel, + 4 smaller
Operator picked six open items to close before the next B4 run (B4 deferred until after the code
changes). All shipped with tests + HTML docs; suite **7770 passed** / 12 skipped / 0 failed. The
measurement discipline changed two diagnoses mid-flight:
- **Off-topic hydration gate (`core/bus.py`, `memory/vector.py`) — the pitched "tune `_RELEVANCE_FLOOR`
  from the ledger" was the WRONG lever, proven by measurement.** RRF scores are a function of (rank,
  tier weight, intent) and DISCARD embedding distance, so normalising against the top item makes the
  best match 1.0 whether the query is on- or off-topic (off-topic scores are actually flatter) — a
  relative floor cannot separate them, and the ledger doesn't carry distance so it can't drive it
  either. Measured the real signal: on BGE-small the best on-topic match is < 0.40, the best off-topic
  match ≥ 0.44 (clean gap). Fix: a `_VECTOR_MATCH_FLOOR = 0.42` best-match gate — if the closest vector
  candidate exceeds it the vector tier injects nothing. On the HYDRATION path only (`search_items(min_
  relevance_dist=…)`); the recall TOOL stays best-effort. On-topic hydration is untouched (true match
  always < gate) so recall can't regress. Flipped the xfail in `test_recall_regression_eval` → real
  pass (verified across 3 off-topic phrasings). `_RELEVANCE_FLOOR` kept at 0.0 as a relative-pruning knob.
- **Streaming tail now stays cancellable (`core/agent.py`) — diverged from the finding's fix sketch,
  with justification.** `handle_chat` returned the stream generator from inside `async with
  agent_semaphore`, so the tail streamed after the outer finally unregistered the turn — invisible to
  `/api/turns`, uncancellable. Fix: the streaming path wraps its generator so `unregister` is DEFERRED
  into the wrapper's finally (runs at drain end), and the stream loop checks `is_cancelled` each chunk
  (cooperative mid-stream stop; finalization still runs on the partial). Did NOT hold the semaphore
  across the drain as the §4B sketch suggested: `stream_chat_completion` already counts
  `foreground_tasks` for the whole stream (the LLM slot isn't stolen), and holding the permit would
  couple turn serialization to CLIENT read speed — a stalled reader would block every later turn, a
  worse failure mode. Live-verified: 4-chunk stream, clean [DONE], 0 active turns after drain.
- **Per-engine Tor search timeout (`tools/search.py`)** — grounded in the recorded 2026-07-08
  measurement (mojeek is the slow-but-reliable winner ~10-18s; others win/fail fast ~1-6s). mojeek
  keeps 18s; fast engines get 12s via `_engine_timeout`, freeing their uncancellable `_RACE_POOL`
  thread ~6s sooner on a blocked wave without costing wins. Marginal (the dedicated pool already
  bounds starvation), but clean and measured.
- **`is_published_port` (`sandbox/services.py` + `docker.py`)** — was consulting the CONFIGURED range,
  so a 2nd instance that published NOTHING (all fixed ports taken) still claimed the port and pointed
  the operator at the FIRST instance's forwarder. `DockerSandbox` now records the set it ACTUALLY
  published at container (re)create (`published_service_ports()`, empty for a 2nd instance) and
  `is_published_port(port, published_ports=…)` treats it as authoritative; None → configured-range
  fallback (single-instance, unchanged).
- **Skill-list truncation (`tools/acquired_skills.py` + `registry.py`)** — asked to "list all skills"
  the model re-tabulated every built-in tool with its full schema and ran out of budget before the
  custom ones (verifier late-refuted). `manage_skills(action='list')` now returns the COMPLETE compact
  inventory in one call — acquired + composed macros — with a footer steering the model to summarise
  built-ins by category, not reproduce schemas; description updated to match.
- **Embedder leaked-semaphore (`memory/vector.py`)** — traced the `resource_tracker: 1 leaked semaphore`
  (441 occurrences in prod stderr) to TQDM: transformers' "Loading weights" bar calls tqdm `get_lock()`
  which creates a multiprocessing RLock (a named posix semaphore) never reclaimed at SIGTERM. Fix:
  `tqdm.tqdm.set_lock(threading.RLock())` at vector.py import — we never drive bars across processes,
  so a thread lock suffices and bars still render. Verified 0 leaked (was 1) under SIGTERM via the real
  module path.

New test files: `test_streaming_tail_cancellable.py`, `test_embedder_semaphore_leak.py`. Deployed
(plain-kill → launchd, pid 96856, health ok); also **backed up the live `rrf/weights.json`** (it held
the OLD buggy-fit contextual row from before today's `fit_intent_weights` fix — `graph 0.224`) →
`weights.json.buggy-fit-bak-20260715` so the agent boots on hand-tuned defaults and the corrected refit
relearns from the kept 297 observations.

### 2026-07-15 — never-reviewed cohorts sweep + image-node auth (route timeout 8s→12s)
Started from a live `verify → Worker Node (Nova): ReadTimeout` at exactly 8.0s. Root cause: a
`route()` call that queues behind TWO worker calls (VERIFY firing alongside the request-start
classifier burst) lands just over the 8s `_ROUTE_TIMEOUT_S`. Raised to **12s** (`core/llm.py`;
still fails fast on a genuinely sick node — breaker trips at 3 strikes). The real lever is still
more worker slots (operator: bump nova's `-np`); this is margin. Tests + docs updated.

Then answered "what haven't we reviewed?" by cross-referencing §5B/§5C/§4 against the module
inventory. Five never-reviewed cohorts: the **2026-07-14 memory upgrade** (biggest un-audited
surface — bus/sessions/rrf_weights/dream write+read paths), the **2026-07-08 `_race_search_wave`**
(only the fetch half was audited on 14i), the **2026-07-13 web-UI JS** rewrite, the clockwork repo
copy, and the post-hunt GAIA/ablation scripts. Ran a 6-cohort Workflow (one read-only finder per
cohort → one adversarial refuter per finding; 39 agents). **32/33 confirmed, 1 refuted**; fixed all
32 with regression tests. The headline defects and fixes are tabulated in `docs/audit_fixes.html`
(new §"Never-reviewed cohorts sweep"). Load-bearing ones:
- **Memory read path (`core/bus.py`, `core/sessions.py`):** the PAST CONVERSATIONS tier surfaced
  the CURRENT session's own history under a "NOT the current conversation" header (now excludes the
  active `session_id`, threaded from the route); the `last_hydration` stash had no turn identity so
  an overlapping/hydration-skipping turn misattributed the usefulness observations that train RRF
  (now `turn_id`-stamped compare-and-consume); **session eviction was dead code** (`_evict` listed
  via the clamped `list()`, slice always empty → unbounded growth; now globs by mtime) and the
  message cap silently stopped truncating with ≥cap systems + accumulated thin-client system dupes.
- **RRF learning loop (`core/rrf_weights.py`, `core/dream.py`):** the fit could collapse a
  hand-tuned hot-path weight 2.0→0.1 from 3 correlated same-turn observations, mapped rate 0.5 to
  1.55 (not base), and anchored on the previously-learned matrix making a floored cell sticky. Now:
  anchored piecewise map (0.5→base), `min_obs_per_cell` 3→20, refit anchors on DEFAULTS; partial
  `weights.json` deep-merges over defaults; the observations-ledger append/trim race is closed with
  a shared `LEDGER_LOCK`.
- **Dream writes (`core/dream.py`):** episode batches were marked consolidated on an UNPARSEABLE
  worker reply (`extract_json_from_text`→`{}` ambiguity) → permanent loss (now requeues unless
  `strategies` present); a synthesis byte-identical to a source fragment shared its md5 id so
  `add()` no-op'd then the source-delete erased the only copy (now skips the delete on id collision);
  provenance JSON was string-sliced → unparseable for ≥12 fragments (now caps the list).
- **Search racing (`tools/search.py`):** race losers only asyncio-cancel while the `ddgs` thread
  runs to the 18s timeout on the SHARED `to_thread` pool → starved every other `to_thread` user
  (now a dedicated bounded executor); co-completed loser exceptions retrieved (no ERROR "never
  retrieved" spam); circuit tags fold in a per-process nonce + time bucket (a retried query no
  longer rides the same dead exits); full-timeout hangs re-bucketed empty→timeout; a 4-5-word
  question no longer "reformulates" to itself.
- **Web face (`interface/static/app.js`):** WebSocket reconnect could orphan a live socket (double
  log processing + multiplying chains) — `connectWebSocket` now coalesces in-flight connects and
  tears down the old socket; the stale ICON_CLASS map (WARN/tool glyphs → "think" floor after the
  wide-base glyph migration) regenerated and pinned by a Python↔JS drift test (`?v=` → 3.8).
- **Scripts:** B4's `_wait_arm_quiet` counted self-play "request finished" markers → the
  timeout-bleed guard proceeded into a still-busy treatment arm (HIGH within the harness); driver
  turns now carry a collision-proof `dv` request-id tag ('v'∉hex, distinct from sub-/sched-/job-)
  and only their own END frames count — needed a symmetric optional `request_id` on trackb2's
  `_post`. GAIA `accuracy` excluded errored tasks from the denominator + `--boot` had no port/
  liveness preflight (would score a stale foreign agent); trackb3 `_learning_artifacts` read
  filenames the agent never writes (`graduated_skills.json`→`auto_skills.json`,
  `composed_skills.json`→`composed_skills/composed_skills.json`); B3/B4 wrote records only at
  end-of-run (crash lost hours — now checkpointed per repeat) and rendered boot-failed arms as
  silent zero-yield (now flagged, keep/flip verdict warns); `bash_top_user` injected a
  non-dominant winner (correct answers scored FAILED — burst 12→30, verified 0 losses/ties/2000 seeds).
- **NOT fixed (deliberate):** the clockwork_ghost repo copy's 6 findings — the live device runs a
  newer `face/` package the repo lacks (the repo copy is superseded/non-functional); fixing it
  belongs on the offline device, not the stale repo copy.

**Image-gen node auth** (closes the §4B "no auth on 0.0.0.0 GPU servers" residual, image half; TTS/STT
skipped — currently offline). The Jetson node bound `0.0.0.0` with no auth while a generation
monopolises the GPU ~30-60s. Added fleet `X-Ghost-Key` auth (`GHOST_API_KEY` env → `~/Data/AI/.ghost_api_key`
→ refuse-to-start; checked BEFORE readiness gates; `/health`+`/ready` stay open); agent stamps the key
only on the `image_gen` httpx pool (`core/llm.py` `node_api_key` from `--api-key`). Deployed the repo
copy (which was STALE — it still had the old Mac SDXL server; the device had the hardened Jetson SD1.5
version, now reconciled + auth added). First restart hit the known NvMap/CUDACachingAllocator OOM from
teardown overlap → hardened the loader into a retry-with-backoff (5×20s) that clears the stale
`_load_error` on success, instead of parking until a manual bounce. Verified live: `/health` open,
`/generate` 401 without key / 200 with, authenticated end-to-end generation returned a 205KB PNG.

New test files: `test_unreviewed_cohorts_20260715.py`, `test_ablation_scripts_20260715.py`,
`test_imggen_node_auth.py`, `test_web_icon_map_drift.py`. Suite **7757 passed** / 12 skipped / 1 xfailed
(`env -u FORCE_COLOR`). **Needs prod restart** for the memory-read/write, RRF, search, and image-pool-key
changes (done: local agent redeployed via plain-kill; Jetson node restarted by operator).

### 2026-07-14m — browser click fail-fast: impossible selectors no longer eat the repair turn
Operator spotted it in the live log: two WebOS repair turns (3D 18:53, 7C 19:03) died at the same
wall — `click .wp-option` → `TimeoutError: Page.click: Timeout 30000ms exceeded` ×2 → no-progress
loop breaker force-ends the turn before the wallpaper fix could be verified. Root cause is
structural, not flakiness: every atomic browser op launches a fresh context and re-navigates (only
cookies persist), so `.wp-option` — which exists only after clicking the Wallpapers icon — can NEVER
appear; page.click waited out the full 30s on an impossible selector and the opaque error taught the
model nothing (it retried variants of the same doomed call; its reasoning never reached for
`interact`, the op that exists exactly for multi-step flows).
Fix (tools/browser.py, embedded runner): `op_click` probes the selector post-goto with
`wait_for_selector(state='attached', timeout=min(8s, timeout_ms))` — absent → fail in ≤8s with an
error explaining the state reset and naming the escape verbatim (op='interact' + actions list).
'attached' not 'visible' so animating-but-present elements still get page.click's own actionability
diagnostics. Outer failure hint + click op description carry the same steer. Runner is rewritten to
the sandbox per call, so this needs only the agent restart. One stale FakePage in
test_browser_navigate_text.py grew a no-op wait_for_selector.
Tests: test_browser_click_failfast.py (4; runner-exec harness). All 237 browser tests + full suite
green. Docs: tools/browser.html (new section). NOT deployed — operator restarts manually.

### 2026-07-14l — autoadvance now consumes the write-time syntax-check signal
First build after the 14j deploy proved the new visibility AND exposed the unwired loop: a 6-task
WebOS autoadvance rewrote index.html 5×, EVERY write result carried `⚠ SYNTAX CHECK FAILED …
Identifier 'WebOS' has already been declared` (the check firing for the first time in prod thanks to
`_find_node`), yet every task closed DONE — `_looks_like_write_error` only reads the result head, and
nothing else consumed the warning. The broken build was only caught when the final turn browsed the
page (the agent then self-healed it — already better than the pre-14j user-paste-back loop).
Fix (coding_executor.py): `_syntax_fail_reason(path, out)` extracts the diagnostic; all three apply
paths (append / full-content write / edits — last edit's result = final on-disk state) return it as
an apply FAILURE. File stays on disk (file_system semantics unchanged); the retry-with-feedback loop
gets the exact line and a steer toward `edits`; exhausted attempts → CodingResult(ok=False) →
`_finalize_coding` marks the task FAILED and stops the batch, instead of stacking features on a file
that doesn't parse. Fails open when no warning present (unknown ext / node absent).
Tests: test_autoadvance_syntax_gate.py (8: extraction, all 3 apply paths, retry-exhaust → honest FAIL,
taint→fix→retry success). Full suite green. Docs: core/coding_executor.html (new section).
NOT deployed — operator restarts manually.

### 2026-07-14k — 14j follow-ups: finish-line honesty guards + inline `-c` AST rescue
Closed the two "observed, deliberately unchanged" items from 14j.
- **Trailing-promise guard (agent.py).** The 14j corrupting turn didn't hit the 40-turn cap — it
  finalized normally on narration ("…That's what's causing the error. Let me fix it.") and the
  conversational-filler guard only fires on tool-NAME mentions. New `_ends_with_action_promise()`:
  last sentence ≤120 chars starting an imminent action (`let me` minus `let me know`, `I'll`,
  `I will`, `I am going to`, `gonna`) after a tool-running turn → ONE act-or-admit steer per request
  (latched like the notify guard; pure conversation exempt via has_run_tools).
- **Dropped-mutation honesty note (agent.py).** force_final_response drops queued tool_calls by
  design (post-terminal-tool hallucinations), but a dropped MUTATING call (file_system/execute/
  manage_services/manage_projects/database — observed 2026-07-12 ×2 eating file_system at the finish
  line) left the reply implying the work ran. `_dropped_mutation_note()` appends "⚠ … has NOT been
  applied yet" to the final reply; terminal-tool drops stay silent.
- **Inline `-c` AST rescue (execute.py).** The auto-convert's quote-safe gate is a proxy for "shlex
  can reconstruct bash's view" — irrelevant to the base64 transport, which never lets bash see the
  body. A long valid-Python body mixing quote types (the 14j 769-char cleanup script, blocked twice)
  now rescues: shlex path unavailable → `ast.parse(raw regex-captured body)` → parses → ship
  byte-exact via base64. Python only, no skill wraps, no trailing pipe; invalid bodies still BLOCK.
  3 stale tests in test_inline_c_guard_cd_prefix.py updated to the new (strictly better) expectation.
Tests: test_pending_action_and_inline_rescue.py (14). Full suite green. Docs: core/agent.html
(finish-line honesty guards), tools/execute.html (AST-rescue section + stale "still blocks" line).
NOT deployed — operator restarts manually (14j + 14k ship together).

### 2026-07-14j — "correct code" failure chain: marker-leak replace parser + 3 compounding guards
Operator report: "when I ask it to correct code it consistently fails — LLM or us?" Verdict: **us.**
Trajectory trawl (394 records, 07-12→14) + live-stream forensics on the WebOS episode showed the model's
diagnoses were correct and canonical single-envelope replaces applied 5/5; the harness did the damage.
**Root bug — multi-edit envelope marker leak (file_system.py aider path).** Only the FIRST `====` in a
`<<<< SEARCH` envelope is the separator; a second edit packed into the same envelope lands in
`replace_str` VERBATIM — extra `====` lines + both texts written to the file, result "SUCCESS".
That's exactly how index.html shipped with `====` at lines 78/80/85 + 3 duplicate `let zIndex`
declarations; the user's next two messages were the browser errors. Four compounding gaps, all fixed:
- **Parse-time rejection:** a block whose replacement contains a bare marker line (`====` exactly-4,
  `<<<< SEARCH`, `>>>>`) is rejected with a one-envelope-per-edit steer (multi-envelope calls unchanged).
- **`_marker_leak` write backstop** in `_write_replace_guarded` (covers exact/flexible/fuzzy/anchor +
  native-args merges): refuses any result that would ADD marker lines vs. prev content. Count-aware so
  cleanup edits on an already-corrupted file still land. RST `=====` underlines don't trip it.
- **js/html syntax-regression rollback:** `_syntax_regression` only covered .py/.json — an HTML-corrupting
  replace was un-rejectable. New async `_syntax_regression_js_html` runs `node --check` on .js and inline
  `<script>` blocks with the same parse→no-parse semantics (fail-open sans node; broken files stay editable).
- **`_find_node()` — the reason the EXISTING post-write html/js check never fired in prod:** launchd PATH
  has no /opt/homebrew/bin, so `shutil.which("node")` was None and checks silently skipped. Now falls back
  to the standard install paths. (Generalize: any which()-based optional checker is dead under launchd.)
**Also: `file_system search` exit-1 misreport.** rg's "no matches" exit 1 came back as the docker layer's
`[SYSTEM ERROR]: Process failed (Exit 1) with no output.` sentinel — the agent couldn't verify its own fix
(3 identical `====` searches burned turns; execute.py had this normalization since the chess session,
search didn't). Ported: exit 1 + empty/sentinel output → "Report: No matches found…"; exit 2 passes through.
Secondary observations from the trawl — BOTH CLOSED same day in 14k (trailing-promise guard +
dropped-mutation note + inline `-c` AST rescue): the corrupting turn ended at n_steps=31 at literally
"Let me fix it"; inline `python -c` guard cost ~4 steps/turn in repair loops. Tests: test_replace_marker_leak_guard.py (16).
Full suite green. Docs: tools/file_system.html (new 2026-07-14 section + ops-table rows).

### 2026-07-14i — search fetch + 4 unreviewed tools (database/report_pdf/image_gen/system)
Two-part audit. Headline finds were anonymity leaks and a DB SSRF bypass.
**Fetch/anonymity (search.py, darkweb_search.py, utils/helpers.py):**
- **DNS leak on Tor page fetch (HIGH for an anonymity tool).** `helper_fetch_url_content` (behind
  deep_research + knowledge_base URL ingest) and darkweb's `_fetch_onion_text` validated URLs with the
  SSRF guard's default resolve=True → host-side getaddrinfo of every fetched hostname, leaking the DNS
  query for the site being visited anonymously. For .onion it leaks WHICH hidden service. Both always
  fetch over Tor (resolution happens at the exit), so the host lookup only leaked. Fixed:
  resolve=not bool(proxy) / resolve=False — mirrors browser/download's resolve=not anonymous.
- `_filter_junk` crashed on a result with explicit href=None → at the try-guarded call site it sank
  the whole engine's result batch. Now `(r.get('href') or r.get('url') or '')`.
- Redirect-not-revalidated in these fetches is Tor-mitigated (exit-node routed, not host-reachable) —
  documented, not a hole in Tor mode. The report_pdf/vision/download redirect fixes were the non-Tor cases.
**database.py (survey + verified):**
- **Host-restriction SSRF bypass via libpq URI query params (MED).** Guard compared urlparse netloc
  only, but `?hostaddr=10.0.0.99` is libpq's actual TCP target (host→SNI); `?host=/?port=/?dbname=`
  also override. Verified with parse_dsn: `…/prod?hostaddr=10.0.0.99` connected to 10.0.0.99 while
  reading as prod. Now compares canonical parse_dsn keys (hostaddr>host precedence). Non-numeric port
  → formatted error (was uncaught ValueError).
- **confirm="false" authorized DROP/TRUNCATE (MED)** — bool("false") is True. Now affirmative-token only.
- schema output row-capped (was unbounded flood); session statement_timeout no longer leaks across
  pooled calls; validator fail-open now logs WARNING not debug.
**report_pdf.py:** files-that-exist hint now fires on the all-source-files-missing error path (was
success-path only — the exact scenario it was built for); hidden-dir filter uses relative parts.
**image_gen.py:** SUCCESS message states actual (snapped) dimensions (was operator-log only → model
lied about size / re-called); mkdir before write (was discarding a GPU-paid image on missing dir).
**system.py:** null-valued profile keys no longer crash location lookup (`(data.get("root") or {})`);
unknown location → clear message not "failed: None"; localhost/::1 count as Tor mode; bare except
narrowed; cpu_percent moved off loop thread. Reviewed residual (documented, NOT changed): verify=False
on the HTTPS checks — flipping on the live Tor path risks regressing weather/health vs quirky exits;
narrow threat model; weather already untrusted content.
Tests: test_fetch_dns_leak.py (6) + test_tools_batch_audit_fixes.py (15); 3 stale tests updated
(schema SQL + fetchmany, one ssrf lambda signature). Full suite green. Docs: 6 tool pages.

### 2026-07-14h — tasks.py + scheduler audit: user cron tasks silently died on EVERY deploy
The predicted "invisible for weeks" bug was real, just not in a formatter: the AsyncIOScheduler
jobstore is IN-MEMORY and deploys are plain kills, so every deploy wiped all user-scheduled tasks —
while the "task X is running" note in vector memory kept asserting they were alive. Nobody watches
scheduled output, so nothing surfaced the loss. Fixed:
- **Persistent task store**: create → $GHOST_HOME/system/scheduled_tasks.json (atomic write, persisted
  only after live registration succeeds); stop/stop_all unpersist; `restore_persisted_tasks()` at
  lifespan start re-registers via the same `_add_job` trigger-builder the create path uses (malformed
  records skipped+dropped, never aborting the rest). No memory_dir ⇒ clean no-op (old semantics).
- **UTC ambiguity**: list output now says "(times in UTC)" and the manage_tasks schema tells the model
  to convert local-time requests ("9am" = '0 6 * * *' in Athens summer). Deliberately NOT switched to
  host tz — existing expressions were authored under UTC semantics.
- Schema also documents persistence + same-name-replaces semantics.
Verified CLEAN (checked, no change needed): should_defer_scheduled_task reads the real request-scoped
`foreground_requests` counter (routes.py increments it around whole user requests — my initial
"wrong attribute" suspicion was wrong, retracted); the 2026-07-11 conclusion-recording path
(record_scheduled_result → activity ledger → digest/push) does deliver task output (300-char digest
by design); the interval-validation and action-normalisation fixes hold; scheduler-error listener
isolation intact. `idle_dream_monitor` list-filter is a harmless vestige (nothing adds that job).
Tests: tests/test_tasks_persistence.py (9); 50 existing tasks/scheduler tests green. Docs:
docs/tools/tasks.html (persistence + timezone sections).

### 2026-07-14g — execute.py audit: cross-scope run-path gap (14c integration), gate parity, retry spill
Final audit of the review sweep (file_system → fact_check/vision/composed_skills → browser → execute).
execute.py was the healthiest so far — its 2026-07-02 chess-trace fixes hold — but the sweep's own
14c change opened one gap, plus three latent issues:
- **Root-anchored run path (14c integration).** file_system's root-anchoring means _get_safe_path can
  now resolve `/workspace/x.py` to the OUTER root under a scoped session; execute's rel_path
  derivation (relative_to(scoped) → lstrip fallback) then minted the phantom `workspace/x.py` →
  ENOENT. Such files now run via container-absolute `/workspace/<rel>` — cwd-independent, same file
  the read/write touched.
- **Both not-found retries dropped `spill_large_output`** (remap + root-cwd) → a noisy retry dumped
  full output into context; and the root-cwd retry succeeded SILENTLY, so the model never learned the
  file lived at the root and re-issued scoped paths. Retries keep spill; root retry announces itself.
- **Script-branch workspace gate was missing** despite the comment claiming parity with the bash
  branch — every fast successful script run was recorded (duration 0.0) into the activity ledger.
  Now timed + gated (failures always, successes ≥5s).
- **Egress-guard bypass on run-existing-file, closed non-blockingly:** a probe script written earlier
  via file_system ran unchecked (guard only vets command/inline content), but hard-blocking would seal
  legit apps that reference the agent's URL by design. The run proceeds with a SANDBOX LOOPBACK BLIND
  SPOT ground-truth note appended when source matches URL+net-client — breaks the mock-server
  misdiagnosis chain with zero false-positive blocking. Also: dead in-function `import json` removed;
  docs/tools/execute.html had two sections AFTER its footer (fixed).
Tests: tests/test_execute_audit_fixes.py (9); all 72 existing execute tests green unchanged.
This closes the systems-review sweep — remaining known threads live in §4B + the 14e ws:// residual.

### 2026-07-14f — first-request latency: main-node prefix warmup (~70s → ~20-25s expected)
Operator report: first request of a session prefills 32-33k tokens ≈ 1 minute (KV cache fine after).
Diagnosis: prefill measured at ~450 tok/s (llama-server.log); the rendered head = system slot
(SYSTEM_PROMPT+profile, 14.3KB) + native tool schemas (63KB, 39 tools — the whale, rendered by the
chat template right after the system text) ≈ 20-24k tokens, and it is BYTE-STABLE across
conversations (continuity blocks were already moved to the tail injection for cache stability;
`working_memory_context` is a vestige, always ""). Divergence only starts at the query-routed
acquired-skill tail + hydrated memory in user msg 1.
Fix: `GhostAgent.warm_up_main_prefix()` — one background max_tokens=1 request at lifespan start
carrying the byte-exact head, built through the SAME code paths as a live request
(_RequestState.get_profile_str → SYSTEM_PROMPT splice + perfect_it mutation; get_active_tool_defs
for tools). is_background=True targets the main slot but yields to any live foreground request;
best-effort (failures debug-logged); opt-out GHOST_MAIN_PREFIX_WARMUP=0. Sibling of
warm_up_workers (2026-07-12, off-main nodes). Expected: first user request pays only its unique
tail (~8-10k tokens ≈ 20-25s instead of ~70s). NOTE: warmup covers agent boot; a llama-server
restart mid-session wipes the cache until the next agent deploy (acceptable — they usually
co-restart). Operator is separately testing --ubatch-size 2048 for raw prefill throughput.
Tests: tests/test_main_prefix_warmup.py (8). Docs: docs/core/llm.html (new section).

### 2026-07-14e — browser tool audit: the text-preview feature never reached the model (formatter dropped it)
Same review-then-fix pass as 14c/14d, over `tools/browser.py` (1895 lines incl. the in-sandbox runner).
- **HEADLINE: navigate/click formatters silently DROPPED the runner's ~8KB text preview.** The
  2026-07-07 nav-preview feature (one op instead of navigate→extract_text, each a full Chromium
  relaunch + Tor re-fetch) was computed and shipped by the runner — and discarded by the host-side
  result formatter. The runner side was tested; the formatter wasn't. `click` also dropped the
  captured JS diagnostics (a click that crashed page JS looked identical to one that worked). Both
  now render a PAGE TEXT block (+ js_errors for click).
- **Render-check false positive**: `analyze_screenshot_render` flagged ANY frame ≥80% one colour as
  BLANK — i.e. every white-background TEXT page. Poisoned evidence invites the verifier to refute
  true "it renders" claims. Uniform now requires few distinct colour buckets too
  ((≥80% AND ≤24 buckets) or ≤6 buckets) — sky/loading frames still flagged, docs pages pass.
- **Interact screenshots bypassed the render check entirely** (only the atomic screenshot op was
  checked) → container→host path map at sanitise time; every interact capture now gets RENDER_CHECK.
- **Dead/broken knobs**: `nav_text_chars` was never plumbed host→runner (the preview size was
  unconfigurable); `post_click_ms` likewise; `settle_ms="2s"` raised a raw ValueError OUT of the tool
  (now `_safe_int`). Schema now advertises settle_ms/click_center/nav_text_chars (previously
  learnable only from error-hint text).
- **KNOWN RESIDUAL (documented, not fixed)**: Playwright's `ctx.route()` cannot intercept
  WebSockets, so the in-runner SSRF interceptor doesn't vet ws:// — exploitable only while a
  supervised service is running (the `<loopback>` proxy bypass opens all loopback ports and the
  port gate doesn't apply to WS). Chromium has no disable-WS flag; documented in
  docs/tools/browser.html with the exposure analysis.
Tests: tests/test_browser_formatter_and_render_fixes.py (9 new); all 204 existing browser tests
green unchanged. Also updated 4 stale fact_check tests + 1 binding-cap test (pinned the pre-14d
behavior — the 14d full-suite run caught them). Docs: docs/tools/browser.html.

### 2026-07-14d — fact_check / vision / composed_skills audit: 1 security hole, 2 correctness bugs, 2 silent truncations, 2 dead-feature closures
Follow-up to 14c: reviewed the three never-reviewed subsystems, then fixed everything found.
- **vision SSRF-via-redirect (SECURITY, `tools/vision.py`).** URL fetches used `follow_redirects=True`
  while validating only the ORIGINAL url — a public page 302-ing to 127.0.0.1/169.254.169.254/LAN
  bypassed the guard. Same hole was closed in tool_download_file 2026-07-07; vision never got it.
  Now: auto-redirect OFF, every hop re-validated via the shared `_download_redirect_target`, 5-hop cap.
- **fact_check returned None (`tools/search.py`).** The old flow forced a deep_research tool call via
  tool_choice just to rephrase the claim; a content-only answer (native-tools corruption family) fell
  off the end of the function → None to the dispatcher. REWRITTEN: deep_research is called directly
  with the claim, then ONE verify call (verdict-first prompt). Kills the None path, the forced-tool
  fragility (empty restricted_tools under subagent allowlists), a whole LLM round, and the
  get_active_tool_definitions rebuild per call. Also: evidence now capped against max_context
  (param was accepted-but-unused), verify failure degrades to PARTIAL + raw evidence, content:null
  coerced (was rendering literal "None"), empty-query guard, error strings instead of raw tracebacks.
  fact_check had ZERO tests → tests/test_fact_check.py (8).
- **vision extract_text_pdf on an image** forced the fitz branch and REPLACED the extracted image data
  with a doomed PDF parse → gated on is_pdf (action is just a prompt choice for non-PDFs). PDF 10-page
  cap now ANNOUNCED (was silent). Local files typed by magic-byte sniff (`_sniff_image_mime`) with
  extension fallback; non-images refused (a .txt was previously guessed image/jpeg and shipped to the
  vision model). prompt aliases healed (question/query/text/instruction); garbage Content-Length no
  longer crashes the fetch. tests/test_vision_hardening.py (11).
- **composed_skills branching was built-but-unwired** — executor honoured branches; nothing could
  author them (define never parsed the fields). Now: define accepts per-step branch_condition/
  branch_target + top-level `branches` dict; validation (sequential-only, targets must exist, dataflow
  checks over branch sequences); branch-only $params mined into the advertised schema; registry schema
  documents it. Plus: define REFUSES an existing name (register() replaced the object and reset usage
  stats — a typo could clobber a tuned macro); save_as bindings >16KB now carry an explicit
  truncation marker (display cap was marked, the BINDING was cut silently); parallel fan-out bounded
  by a 4-slot semaphore (single-slot llama box); `_step_result_ok` counts SYSTEM INSTRUCTION:/REJECTED:
  prefixes (file_system hard failures) as failures — they inflated success_rate; dead `find_matching`
  removed (zero runtime callers). tests/test_composed_skills_fixes.py (9); 4 obsolete tests removed.
Docs: docs/tools/search.html, vision.html, composed_skills.html; registry schema for
manage_composed_skills. Full suite green.

### 2026-07-14c — file_system audit: project-scoped root blindness fixed (the "sandbox is EMPTY" lie)
Operator report: `list_files` "not working for some subdirectories" — the agent gave up on file_system
and explored via `execute` instead. The live log (sessions 1D/77) showed the real shape: with a project
active, an `execute`-side `git clone` to an ABSOLUTE `/workspace/analysis/...` path landed at the
sandbox ROOT while every file_system op stayed scoped to `projects/<id>/` — list said EMPTY, reads said
"does not exist", and the explicit `/workspace/...` form was *healed into* the scoped dir (phantom
path). vision/browser/agent-core each already hand-rolled a root fallback; file_system itself was the
one subsystem without it. All fixed in `tools/file_system.py` (+ qwen_bridge/registry):
- **`list` ignored `path` entirely** (dispatcher dropped it) + silent 200-entry truncation with
  nondeterministic (unsorted) walk order → subdir listings now work, output is sorted/deterministic,
  truncation REPORTS the hidden count and the recovery (`list` a subdir), and an empty scoped listing
  names root-level files with their `/workspace/...` paths. Aliases `ls/dir/tree/list_dir/list_directory` healed.
- **Existence-aware root anchoring in `_get_safe_path`**: `/workspace/X` under scoping resolves scoped
  when the scoped copy exists (historical heal, still the default for new files — write/execute
  symmetry and the browser heals unchanged), to the ROOT when the file/tree genuinely lives there.
  Host-absolute paths under the outer root map the same way. Found+fixed a latent bug in the
  host-absolute branch: `relative_to()` raised on the first non-matching base and aborted the mapping.
  Destructive guard (`allow_root=False`) now also refuses the OUTER root (`delete '/workspace'`).
- **Read-only root fallback** (`_scoped_root_fallback`): read/inspect/read_chunked/search/find/list
  serve a root copy with a NOTE naming its `/workspace/...` path; `replace` deliberately does NOT
  silently cross scope — it returns the exact path to re-issue with. `_missing_file_message` appends
  the root-files hint instead of a dead-end "EMPTY".
- **search/find default scope**: rg/find run at the container root, so the old literal `.` swept the
  WHOLE sandbox (other projects included) when scoped → default is now the active workspace's container
  path (`/workspace/projects/<id>`).
- **qwen_bridge never passed `sandbox_manager`** → search/find on that runtime always died with
  "'NoneType' has no attribute 'execute'"; also now forwards `max_context`/`read_budget`; enum gained `find`.
- **Smaller:** dispatcher dropped `inspect`'s `lines` arg (now forwarded + coerced); httpx download
  rotated the Tor identity TWICE per 401/403/503 attempt (now once); `_syntax_feedback` read files
  with the process locale (LANG unset under launchd → UnicodeDecodeError → check silently skipped) —
  now explicit UTF-8 like the write path.
Tests: `tests/test_file_system_scoped_root_and_list.py` (27 new); 3 stale assertions updated
(search/find "." defaults). Full suite green. Docs: `docs/tools/file_system.html` (new 2026-07-14
section), `docs/tools/qwen_bridge.html`.

### 2026-07-14b — Bug-hunt pass over the never-reviewed cohorts (post-July-hunt shipping since 2026-07-05)
The July static/functional hunts (§5B/§5C) + the 2026-07-07 six-agent review covered everything that
existed THEN; ~10 days of shipping since (host services, notifications, delegation, sessions/cancel,
games, RAG, challenge_templates) had never had a review pass. Ran 4 parallel read-only review agents
(one per cohort), verified every finding against the code myself, fixed the CONFIRMED bugs with tests
+ HTML docs, logged residuals to §4B. **Two HIGH containment breaks were the headline** — a delegated
sub-agent could escape isolation ENTIRELY:
- **Sub-agent tool containment (HIGH, `core/subagent.py` + `core/agent.py`).** The old restriction
  filtered only the dispatch DICT and left `_subagent_allowed_tools` UNUSED, so (a) the SCHEMA the
  model saw was the full registry — it was literally shown `delegate`/`jobs`/`manage_*` and invited to
  call them (recursive fan-out, daemon scheduling, profile writes), and (b) any dispatch miss HEALED
  `available_tools` back to the full registry, undoing the filter. Fixed with three gates: `disabled_tools`
  = advertised − allowlist (filters schema AND blocks dispatch by name), narrowed dispatch dict, and
  `_rebuild_available_tools` re-narrows to the allowlist on every miss. Tests: test_subagent_containment.py (7).
- **Read-only memory façades didn't block the real mutators (HIGH, `memory/readonly.py`).** The no-op
  method names were GUESSED (`add_memory`, `delete_memory`, `add_triplet`, `insert_fact`) and don't
  exist on the stores; the REAL mutators (`add`, `ingest_document`, `add_triplets`, `delete_by_target`,
  `remove_by_trigger`, …) and the raw `.real`/`.collection`/`.nx_graph` handles passed straight through.
  Rewritten as a default-allow-reads / explicit-deny-writes proxy pinned to real method names, `search`
  forced `record_retrievals=False`, raw handles blocked. A mutator-list guard test introspects each real
  store and fails if it grows an unblocked writer. Tests: test_readonly_memory.py (14).
- **Other CONFIRMED, fixed:** swarm jobs landed `str(True)` as their result while content sat in the
  scratchpad → `Job.result_resolver` (jobs.py) reads `output_key` so `collect` returns content; turn
  registry used the client-supplied `X-Request-ID` as key with unconditional overwrite + key-based
  unregister → cross-turn mis-cancel (cancelling B killed running A) → `register` uniquifies on
  collision, `unregister` identity-checked; `/notifications/ack` stored an unbounded watermark →
  permanent consumer wedge → clamped to `[0,EOF]`; notify egress LAN-suffix list diverged from the
  egress guard → `.home`/`.arpa` pushes silently dropped under mandatory-tor → share the guard's
  constants; notify_tool rate-limit consumed a slot before the write → split check/commit; `send_soon`
  task GC-drop → retained; Tor CONTROL port 9051 added to sandbox BLOCKED_PORTS (SSRF defence-in-depth);
  `extract_move_text` IndexError → HTTP 500 on a whitespace-only reply → guarded; self-play
  concurrency-cancel template budget 2.0s couldn't reject a non-cancelling solution (parallel losers
  ~1.5s) → 1.0s; SQL group-by validator sorted by raw float → `-round(t,2)`.
- **Verified HOLDS (agents tried and failed to break):** notify.py Tor fail-closed (public target with
  no proxy raises PermissionError before any socket; no cleartext fallback); browser SSRF guard
  (integer-IP / 0.0.0.0 / redirect / DNS-rebind all blocked); the turn cancellation state machine on
  this interpreter (semaphore release on hard-kill, CancelledError propagation, queued-turn kill, no
  registry leak). httpx 0.28 `proxy=` kwarg is correct; no codebase misuse of the removed `proxies=`.
- **Residuals** (analysis in §4B): streaming-tail-outside-semaphore (HIGH, architectural — deferred to a
  focused turns/cancel session), `is_published_port` multi-instance (MED), `read_since` shrunk-ledger
  (LOW latent), tic-tac-toe parity load (LOW), `_invoke_template` TypeError catch (LOW).
- New tests: test_subagent_containment.py, test_readonly_memory.py, test_turn_registry_collision.py,
  test_bughunt_20260714.py (+1 source-inspection assertion updated for the identity-checked unregister).
  Docs: core/delegation.html, memory/readonly.html (new), core/sessions.html, sandbox/services.html,
  core/autonomous_activity.html, api/game_routes.html, core/challenge_templates.html. Suite **7565
  passed / 12 skipped / 1 xfailed** (`env -u FORCE_COLOR`). **Needs prod restart** for the containment
  fixes, turn-registry, ack clamp, and the notify egress/rate-limit fixes.

### 2026-07-14 — Memory-system upgrade: three unwired loops closed, NapMem-inspired structure, usefulness feedback loop, recall eval harness
Triggered by a comparison against arXiv 2607.05794 (NapMem — "memory as an action space": linked
multi-granularity pyramid + active navigation). Verdict: this agent had ~80% of the storage parts
(and decay/episodic/graph tiers the paper lacks) but sat on the paper's "passive retrieval"
ablation side, and three built subsystems had no caller. Eight items shipped, ordered by the
"close loops before new modules" principle:
- **Scratchpad persists in prod** — `main()` now builds `Scratchpad(persist_path=memory_dir/
  "scratchpad.db")` (plain-`kill` deploys wiped working state incl. the `__current_project__`
  resume sentinel; the main.py:417 NOTE anticipated exactly this flip). `--no-memory` stays
  in-memory. Plus the §4B nit: every sqlite connect wrapped in `contextlib.closing` (the `with
  sqlite3.connect()` form only scopes the TRANSACTION), and a corrupt DB at boot degrades to
  in-memory instead of respawn-looping launchd. Tests: test_scratchpad_persistence.py (+2).
- **Episodic consolidation wired** (`Dreamer._consolidate_episodes`) — `get_unconsolidated`/
  `mark_consolidated` finally have their caller: runs after journal drain and BEFORE the REM
  entropy gate (a thin auto pool can't starve it), one worker call generalizes ≤40 episodes
  (trigger → action chain with FAILED markers → outcome) into imperative strategies through the
  actionability gate, `source="episode"`. Failure contract mirrors the smart-memory requeue fix:
  mark only after a successful parse. This is also the trajectory-shaped seed source §4A(c) said
  dream needs. Tests: test_dream_episode_consolidation.py (8).
- **Graph compression wired** (see §4B item, now closed) — safe/fuzzy candidate tiers, worker
  confirmation, 8-merge/cycle cap.
- **Provenance on abstractions** (NapMem's falsifiability idea) — syntheses store
  `provenance=[{id, excerpt}]` captured BEFORE the merged sources are deleted; lessons carry
  `source_refs` (e.g. `ep:12`, unioned on dedup, mirrored to the vector twin); `tool_recall`
  renders EVIDENCE lines. Tests: test_memory_provenance.py (8).
- **Sessions became the raw-conversation tier** — `SessionStore.search_messages` (50 most-recent
  sessions, mtime-cached parses, 2s summaries memo shared across the RAG-fusion fan-out,
  ≥2-distinct-term floor) feeds a FIFTH MemoryBus fetcher under a PAST CONVERSATIONS header;
  intent weights extended (session: factual .8 / procedural .5 / contextual 1.2, mirrored in
  rrf_weights defaults). Sessions were durable since 2026-07-11 but replay-only — the lowest
  abstraction layer was invisible to retrieval. Tests: test_session_hydration_tier.py (11).
- **Iterative recall** — `knowledge_base(action='expand', ref='ep:12'|'session:<id>')` resolves
  EVIDENCE REFS to raw records (the query_document read→refine loop generalized to memory);
  recall's zero-hit reply nudges ONE reworded retry; §4C recall-routing variance RESOLVED (see
  item). Tests: test_iterative_recall_expand.py (12).
- **Usefulness feedback loop closed** — `_credit_surfaced` was circular (credit for ENTERING the
  prompt). Now `hydrate_context` stashes survivors; both finalization paths spawn
  `judge_hydration_usefulness` (worker, off critical path): used vector items get `bump_helpful`
  (helpful_count weighs 2× retrieval_count in the spaced-repetition half-life), used skills get
  `record_helpful_retrieval`, every survivor appends `(intent, source, used)` to
  `rrf/observations.jsonl`; dream's new `_refit_rrf_weights` fits ≥30 observations, persists
  `rrf/weights.json`, hot-swaps the live matrix, trims the ledger. The learned RRF matrix is now
  an ONLINE loop keyed to real usefulness. Tests: test_hydration_usefulness_loop.py (14).
- **Recall regression eval** (`test_recall_regression_eval.py`) — golden corpus across all five
  tiers through the REAL pipeline (BGE + Chroma + fusion): measured paraphrase recall 100%
  (floors at 75%), all per-tier cases pass. One deliberate `xfail` pins a MEASURED gap: with
  `bus._RELEVANCE_FLOOR = 0.0` an off-topic query hydrates most of a small store — tune the floor
  from the usefulness ledger, not by guessing; the xfail flips visible when it lands.
- NOT adopted from the paper (deliberately): RL-trained navigation policy (their biggest ablation
  win, but their own 9B-without-RL scored BELOW passive baselines — structure alone only pays at
  ~400B prompted scale; no training infra here) and multi-hop navigation on the hot path (each hop
  is a full local-inference round-trip + busts the KV-cache-stable injection).
- Suite **7523 passed / 12 skipped / 1 xfailed (deliberate)** in 3m05s. NOTE for test runs: a
  shell with `FORCE_COLOR` set fails `test_thinking_loop_guards` (env-sensitive, documented in
  run-and-test-setup memory) — run `env -u FORCE_COLOR`. **Needs prod restart** to pick up:
  persistent scratchpad, session tier, judge hook, dream steps.

### 2026-07-13 (later 8) — RAG overhaul: the full PostgreSQL manual (3,075 pages) is now queryable, 6/6 on eval
- Operator goal: "load the full PostgreSQL manual (~15MB PDF) and ask questions". Analysis found the
  store held **160 fragments and ZERO documents** — the doc path had never been exercised — and that
  the manual would fail at the first step. Three phases, all shipped.
- **Phase 1 — streaming, structure-aware ingest** (`memory/pdf_ingest.py`, new). The old path
  **hard-refused >1000 pages** (manual = 3,075), then **silently truncated at 5M chars** (manual ≈
  10M), and materialised the whole text + whole chunk list + an enriched COPY in RAM. Now: pages
  stream one at a time, accumulate per TOC SECTION, chunk, and flush in 256-chunk batches (peak RAM
  = one batch). Caps → 6,000 pages / 40M chars; chunk 600 → **1,200** (600 shredded parameter
  descriptions). **TOC breadcrumbs** are the big win: PDF text has no markdown headers, so
  `semantic_split_text`'s header-prepending NEVER fired and a `wal_level` chunk had no idea it lived
  under "19.5. Write Ahead Log". `build_page_breadcrumbs` walks PyMuPDF's outline with a level stack
  (pops correctly on siblings/new chapters) → every chunk's EMBEDDED text carries
  `Part III › Chapter 19 › 19.5. Write Ahead Log › 19.5.1. Settings`.
- **Phase 2 — the missing loop.** There was NO document-scoped retrieval: the only path to the model
  was ambient hydration (MemoryBus, 4 tiers, RRF, shared 6-12k char budget, ~12 fragments from the
  WHOLE store) — useless against a 3k-page manual. New `VectorMemory.search_document()` (Chroma
  `where={"source": f}`, 60-candidate pool, BM25 rerank, NO priority tiers / time decay / distance
  gate — all meaningless or harmful inside one document) + `knowledge_base(action="query",
  filename=, question=)` returning ranked passages as TOOL OUTPUT the model iterates on. System
  prompt + `recall`'s description now steer document questions here.
- **Phase 3 — embedder swap.** all-MiniLM-L6-v2 → **BAAI/bge-small-en-v1.5** (also 384-d, so the
  Chroma schema is unchanged). MiniLM is SYMMETRIC-similarity trained with a 256-token window and
  weak on technical/SQL text; doc QA is ASYMMETRIC — its exact failure mode, which the code conceded
  by relaxing the document threshold to 1.25 "for Asymmetric QA". BGE takes a query instruction,
  applied to QUERIES only via `embed_query` (Chroma's `query_texts=` reuses the DOC embedder and
  can't express the asymmetry → the doc-QA path embeds the question itself and passes
  `query_embeddings=`). **Silent-wrongness guard**: both models are 384-d + L2-normalised, so a swap
  raises NOTHING — the vectors just stop meaning anything. Store now carries an `embedder.json`
  fingerprint; boot REFUSES on mismatch and points at `scripts/reembed_memory.py` (snapshot → JSONL,
  recreate collection, re-add; 161 fragments in 0.7s). **The guard fired for real** during the
  migration — launchd's respawn hit the un-migrated store and was correctly refused.
- **Measured live**: manual ingested in **2m29s → 3,075 pages, 1,897 sections, 7,131 chunks**.
  6-question eval (wal_level values, VACUUM vs VACUUM FULL, range-partition syntax, pg_stat_activity,
  MVCC isolation levels, shared_buffers) → **6/6 correct section retrieved**, agent citing breadcrumbs
  (`Chapter 27 › 27.2.3. pg_stat_activity`). Unprompted routing verified: a plain "what does
  archive_command do?" made the agent pick the scoped tool on its own and answer with correct
  `%p`/`%f` semantics. Suite **7449 passed / 12 skipped / 0 failed** (new: `tests/test_rag_document_qa.py`
  ×20; updated: the PDF-extraction test now pins the streaming contract, the chunk-id test the
  batch-safe id, the prompt test the new routing). Docs: `memory/vector.html`.
- Ops note: ingest/migration require the agent STOPPED or driven through it (Chroma is single-writer);
  `scripts/reembed_memory.py` refuses to run against a live agent. BGE model is now HF-cached (needed
  before boot — `--mandatory-tor` is fail-closed).

### 2026-07-13 (later 7) — web face: immersion dive ("the grid swallows the camera" while working)
- Operator idea, built after an explicit feasibility pass: while a USER request is in flight the
  camera is swallowed INTO the node cloud (scene scale ×1.7 + camera dolly z 5.0→1.3), drifting
  back out organically on completion. All in `interface/static/matrix_graph.js` (+1-line dev hook
  in app.js); backup at `matrix_graph.js.bak-20260713-preimmersion` (operator-requested).
- Design decisions that make it work: (1) driven ONLY by `workingState` — background/idle activity
  deliberately does NOT engulf (it would fire all night; the swallow means "working for YOU");
  (2) asymmetric ease much slower than workingState (~5s in / ~10s out) so short requests just
  lean — no yo-yo; (3) NEAR-CAMERA FADE added to both shaders (`smoothstep(0.3,1.4,-mvPosition.z)`)
  so nodes dissolve instead of exploding into screen-filling quads at the camera plane; (4) bloom
  damped ×0.65 fully inside (reply text is read on top of near geometry); (5) look-target blends
  from origin to forward-through-the-cloud (`lookAt(0,0,-3.5·dive)`) to dodge the lookAt-origin
  singularity that turns parallax into wild rotation; (6) reduced-motion caps the dive at 0.15.
- Verified live headlessly via new `window.__ghostFace.getDebugState()` hook: idle camZ=5.00/scale
  0.90 → swallowed camZ=1.33/scale 1.54 → released camZ=4.94/scale 0.92, zero page errors, and the
  swallowed screenshot reads as genuinely inside the lattice with UI text still legible. Cache-bust
  → v3.5. Interface tests 60 passed (new pins: `tests/test_interface_face_immersion.py` — guard the
  working-state-only trigger, reduced-motion cap, near-fade, bloom damping, lookAt blend, backup).
- **Interior enrichment (v3.6, same day)** — operator: "fully zoomed it looks kind of empty".
  Root cause: the scale swell DILUTES local node density exactly when the camera is closest. Three
  dive-gated compensators (resting view untouched): (1) **interior motes** — 400 (150 mobile) tiny
  jewel-tinted particles with shader-side drift (zero per-frame CPU, one draw call,
  `visible=false` at rest), inward-biased distribution along the camera's path; (2) **thicker
  web** — proximity threshold eases up ×1.5 with the dive (O(n²) distances computed anyway; only
  accepts more pairs, MAX_LINES-capped); (3) **faster data pulses** inside (+60% line-pulse tempo)
  + scale boost trimmed 0.7→0.55. Headless re-verify: interior now dense (thick web + junction
  stars + mote haze), no page errors. Interface tests 61 passed.
- **Brightness/density tuning (v3.7, operator: "too many lines and too bright")**: web-thickening
  +50%→+15%; bloom damping inside 0.35→0.5; NEW per-line `diveDim` (×0.7 fully inside) — additive
  stacking of overlapping lines in front of the camera was the real brightness driver. Headless
  re-verify clean; pins updated.
- **Glass chat + translucent log drawer (style.css v3.2, operator: "chat panels hide the face")**:
  bubble fill alphas cut (agent 0.45→0.20, user 0.62→0.28, system 0.5→0.3) with blur strengthened
  (15→18 / 12→16px) — the blur carries readability, the fill was only occluding the face; log
  drawer 0.92→0.58. Verified headlessly with injected large-output bubbles over the busy face.

### 2026-07-13 (later 6) — worker-node model bake-off: Ornith-9B REJECTED, Gemma 4 E4B stays (2× faster AND more accurate)
- Operator was seeing ~20-30 t/s on nova (worker) and asked for a model recommendation. I researched
  and recommended **Ornith-1.0-9B-heretic-MTP** (Qwen3.5-9B lineage, MTP head, strong sub-10B
  benchmarks). Operator swapped it in; we then benched it properly and **the recommendation was
  WRONG on both axes**. Reverted to Gemma 4 E4B.
- **Speed**: Ornith 12–18 t/s decode vs Gemma **18–34** (wall-clock: decompose 9.5s vs **4.5s**;
  web-summary 7.7s vs **3.5s**). Root cause: nova is a base M4 (16GB, ~120GB/s) and decode is
  BANDWIDTH-bound. Gemma E4B's 33 t/s on a 5.1GB file would require 168GB/s — impossible — proving
  the E-series streams far less than its file per token (per-layer-embeddings / MatFormer =
  genuinely "effective 4B"). Ornith streams its full 5.9GB → ~14-20 t/s ceiling, and it was AT that
  ceiling. My earlier claim that "E4B decodes at its raw size" was simply false.
- **Quality (the real surprise)**: on the two bench tasks with a checkable answer, **Gemma WON**.
  Verify: given a deliberately under-supported claim (exit 0 + stdout `42` ≠ "printed the 9th
  largest"), Ornith wrongly **CONFIRMED** (fail-open — precisely what the verifier exists to
  prevent) while Gemma correctly **REFUTED with the reason**. Difficulty-classify: Gemma right
  (`advanced` — the prompt described the malformed-lines twist), Ornith wrong (`basic`). Tied on
  decompose / web-summary / memory-extract / Greek. Benchmark reputation (MMLU-Pro, GPQA) measures
  long-form reasoning, NOT this workload of short mechanical calls.
- **Three dead-end hypotheses, all measured and killed** (documented so nobody re-runs them):
  (a) *swap thrash* — RSS was 9.9GB/16GB with 9.5GB swap used, but `vm_stat` **pageouts did not
  move during the bench**: stale swap from the model transition, not live paging; (b) *oversized
  KV* — dropping `--ctx-size` 131072→65536 freed 1.5GB RSS and changed speed by **zero**;
  (c) *under-drafting* — `--spec-draft-n-max` 3→6 made it **worse** (acceptance collapsed 0.9→0.28;
  6 draft passes to keep ~2.5 tokens on a compute-limited box). Deeper speculation is not free.
- **Live config restored = the ORIGINAL config, which was already optimal**: Gemma 4 E4B UD-Q4_K_XL
  + mmproj + MTP draft, `--ctx-size 131072` (its KV layer-sharing makes that affordable — Ornith's
  doesn't), `-np 4`, **`--spec-draft-n-max 2`**, RSS 7.4GB, no swap pressure.
- **Draft-depth A/B settled (same model/ctx, only n-max varied)**: **n-max 2 WINS** — mean 21.4 t/s
  vs 19.6 at n-max 3 (decompose 3.87s vs 4.59s, ~16% faster; server decode 23–36 vs 18–34 t/s), and
  per-token acceptance is HIGHER at depth 2 (0.61–1.00 vs 0.53–0.76). Combined with Ornith's n-max 6
  collapse, the rule for this box is settled: **shallower speculation wins on a bandwidth/compute-
  limited M4** — each extra drafted token costs a full draft forward pass and acceptance decays with
  depth. My "raise n-max" hypothesis was wrong on BOTH models.
- **Net outcome of the whole exercise: nothing was broken and nothing needed changing.** The operator's
  original 20–30 t/s is this hardware's honest ceiling for this model, and the config they already had
  was the best of everything we measured.
- **Bench gotcha worth keeping**: every candidate is a thinking model — a raw bench MUST send
  `chat_template_kwargs:{enable_thinking:false}` or the whole budget goes to `reasoning_content`
  and `content` comes back EMPTY (12 t/s and blank outputs). The agent's worker path already does
  this (`llm._disable_thinking`, added 2026-07-11 for this exact reason).

### 2026-07-13 (later 5) — web UI: live log console (header button → bottom drawer)
- Operator asked for a button showing the logs in near-realtime. The transport already existed —
  the interface's WebSocket has streamed the pretty log to the browser since 2026-07-11 (it drives
  the face envelope + planner monologue); it just wasn't rendered anywhere readable. Front-end-only
  change (`interface/static/`): `#logs-btn` (terminal icon, header) toggles `#log-console` (bottom
  drawer, 44dvh, render-window visual language). 500-entry ring buffer fed in `ws.onmessage`
  UNCONDITIONALLY (collects while closed → opening shows history); ANSI stripped client-side; lines
  dim monospace with the face's icon→jewel-accent mapping as a left border, errors crimson;
  tail-following auto-scroll with a "paused — N new" pill when scrolled up; DOM capped at buffer
  size. Cache-bust: app.js+matrix_graph → v3.4, style.css → v3.1 (index no-cache, plain reload).
- Verified live headlessly: clicked the button, appended a marker line to the real agent log
  (tail -F broadcast it), marker rendered in the open drawer; history survived close/reopen; no
  page errors. Interface tests 52 passed (new pins: `tests/test_interface_log_console.py`). Docs:
  `interfaces/web_server.html`. No server restart needed (statics read from disk).

### 2026-07-13 (later 4) — Slack notifications were DEAD for 2 days: pipeline wedge (both halves) + finish-line guard
- Operator report: "notify me in slack when done" produced nothing (reqs `11fe11d8`, `bebd549d`).
  Three distinct defects found; all fixed, tested, deployed, and proven live.
- **(1) Delivery pipeline WEDGED since Jul 11 15:14.** Two interlocking halves:
  *Server* (`routes.py::notifications_pending`): `read_since(limit)` bounds SCANNED LINES, not
  returned records — from the stale watermark (3774) every 20-line window was all info-severity
  (dream/self-play spam), so every poll returned `[]` while notify records sat beyond the window.
  *Bot* (`slack_bot/main.py`): the poller only ACKed non-empty responses, so the empty-response
  watermark never persisted → same window re-scanned every 30s forever (thousands of identical
  200-OK polls in the bot log; `notify_consumers.json` mtime frozen at Jul 11). Victims: "Meta
  project complete", a needs-your-input, and both of the agent's own test notifications. FIX:
  pending now scans forward in 200-line chunks (≤50/poll) to `limit` notify records or EOF
  (`limit` soft — whole chunks kept, else the watermark would skip past unreturned records); bot
  acks EVERY response with a watermark (still after delivery — at-least-once preserved) and logs
  `delivered N notification(s)`. Poller body factored into testable `poll_and_deliver_once()`.
- **(2) req 11fe11d8: the agent PLANNED the notify call and never made it.** Research + plan + PDF
  all done, reasoning said "now send the Slack notification", then the final response shipped with
  zero `notify_operator` call — and the verifier CONFIRMED (the deliverable itself was fine). FIX:
  finish-line guard in the turn loop — `_user_asked_for_notification()` (narrow: "notify/ping/alert
  me", comm-verb+slack-destination in one clause, "send me a notification"; negations and questions
  ABOUT slack don't arm; 4k truncation so pasted docs can't) + one-shot SYSTEM-ALERT steer when a
  turn finalizes with the ask unfulfilled. Never fights force-finalisation.
- **(3) req bebd549d burned 17 turns diagnosing INSIDE the sandbox** (printenv/ps/find for the
  slack bot — invisible from a container by design; the loop-breaker fired twice). Not separately
  fixed: the honest limitation is documented; with (1)+(2) fixed the situation shouldn't recur.
  Classic [[sandbox-loopback-blind-spot]] shape, worth remembering.
- **Live proof after deploy**: pending surfaced the 4 stranded records; bot delivered all 4 in ONE
  DM (`delivered 4 notification(s) → U56CVBHHQ`); watermark 3774→40308 (first movement in 2 days);
  queue drained. Suite **7412 passed / 12 skipped / 0 failed** (new: pending-scan tests, poller
  ack-contract tests via fake httpx, `test_notify_finish_guard.py`). Docs:
  `core/autonomous_activity.html` (wedge + guard sections).

### 2026-07-13 (later 3) — API auth ENABLED everywhere: key minted, rotated, rolled to every client
- Closes BOTH standing security flags at once: the agent's `--api-key ""` on a 0.0.0.0 bind (boot-log
  SECURITY WARNING, flagged in the overnight review) AND the interface's publicly-known
  `ghost-secret-123` (flagged 2026-07-12). One shared secret now guards both.
- **Canonical secret file**: `~/Data/AI/.ghost_api_key` (openssl rand -hex 32, mode 600). ONE file
  to rotate; every launcher reads it at start.
- **Agent** (`bin/start-ghost-agent.sh`): exports `GHOST_API_KEY` from the file (env, NOT argv — the
  secret stays out of `ps`); `--api-key ""` removed from the exec line. Missing file fails OPEN with
  a loud log line (a refusing boot would respawn-loop under KeepAlive) — treat that line as a page.
- **Interface** (`bin/start-ghost-client.sh`): exports the same key, deliberately OVERRIDING the
  stale `ghost-secret-123` in `/Library/LaunchDaemons/com.local.ghost-client.plist` (plist edit
  needs sudo; user-owned script is the override point). Front door verified: old key → 401, new →
  200. Upstream baked key = agent key → proxy chain works unchanged.
- **Slack bot**: `.env` key set (was explicitly empty for the authless agent), chmod 600; rebooted;
  poller confirmed 200 against the authed `/api/notifications/pending`. `.env.example` + docs
  updated (the "leave EMPTY for prod" guidance is now wrong and says so).
- **`ghost` CLI** (`bin/ghost`): default key now env → secret file → "" (retired ghost-secret-123
  fallback). **Scripts** (gaia/ablation/claude_trainer) already read `GHOST_API_KEY` env — export
  from the file when driving prod.
- **uConsole client** (`interface/externals/clockwork_ghost/client.py`): the four hardcoded
  `"YOUR_KEY_HERE"` headers (worked only because auth was off) replaced with `_resolve_ghost_api_key()`
  (env → `~/.ghost_api_key` on device → `.ghost_api_key` beside client.py). **DEPLOYED same day**
  once clockworkpi came up: device copy backed up (`~/backup/client.py.pre-apikey-20260713`), merged
  client pushed to `~/bin/client.py`, key at `~/.ghost_api_key` (600), client restarted via
  `setsid launch_ghost.sh` (DISPLAY=:0), device→agent verified 200-with-key / 403-without, no
  Auth-Rejected lines on the agent. **Repo↔device drift caught during the diff**: the device copy
  had dropped the pinned `"model": "qwen"` from the chat payload (pinning 404s ModelNotFound on
  every model upgrade) — the repo copy was STALE and would have reintroduced the bug; fix
  backported to the repo BEFORE deploying. Lesson: always diff a device-deployed external before
  overwriting it.
- **Verified live**: agent 403 without key / 403 with old public key / 200 with new key; fresh boot
  has NO security warning; web UI end-to-end (Playwright: SYSTEM ONLINE, green dot = page → proxy →
  agent stream all on the new key); Slack bot clean boot. `/api/health` now REQUIRES the key —
  update any ad-hoc curl habit (§2 examples updated).
- Left as-is, deliberate: 0.0.0.0 binds themselves (LAN/tailnet reachability is the point — auth is
  now the gate); the stale plist env value (harmless: overridden, and the old key is dead); voice
  services on disorder:8000 (different host/service, no agent auth involved).

### 2026-07-13 (later 2) — web face re-themed: dark-but-MULTICOLOR jewel wheel (operator: "dull")
- Operator liked the 2026-07-12 animation/envelope rework but found the muted near-black palette dull.
  Requirement: "dark but multicolor". Redesign in `interface/static/matrix_graph.js`:
- **5-stop jewel wheel instead of one active hue.** `COLORS.palette` = deep violet `#3e187a` /
  electric blue `#1f39a1` / teal `#0a6675` / emerald `#0f7143` / magenta `#80198f` (all tuned dark —
  additive blending + bloom lift them; warm hues deliberately absent so crimson ERROR stays unique;
  dimmed ~18% from the first cut on operator feedback "a tiny bit too bright" — render check: lit
  fraction 0.386→0.292, white-clip 42→28 px; `COLORS.palette` is THE brightness knob, everything
  else scales off it).
  Each node gets a stable wheel position via an `aSeed` InstancedBufferAttribute; a `uHueDrift`
  uniform slides the whole wheel (~50s/cycle idle, ~15s busy, damped under reduced-motion). Each
  LINE gradients between its endpoints' hues (per-vertex `aLineHue` written in the per-frame line
  builder from `nodeSeeds`). Dim floor keeps a whisper of each hue so the idle graph is multicolor,
  not grey. `uActiveColor` uniform + hardcoded shader companion hue removed. Animations, envelope,
  accent-mood tint, error tint, bloom formula: all untouched.
- `app.js` `_ICON_CLASS_COLOR` mood accents enriched to matching jewel tones; cache-bust bumped
  (app.js + matrix_graph.js → v=3.3 after the dim pass; index.html serves no-cache so a plain
  reload picks it up — NO server restart needed, statics are read from disk per request).
- **Verified in a real headless render** (Playwright chromium + swiftshader against the LIVE :8080
  server, key pulled from the running process env): no GLSL/shader errors; screenshot pixel analysis
  → 5 distinct hue buckets present, lit fraction 0.386 (dark preserved), 42/64k white-clip px
  (bloom cores only). Palette contract pinned by `tests/test_interface_face_palette.py` (5 distinct
  stops, darkness cap ≤0xd0/channel, wheel+attributes wired, app/matrix cache-bust versions move
  together). Interface test set 45 passed. Docs: `docs/interfaces/web_server.html` static-assets
  section rewritten (old "electric-blue↔cyan" description was stale).

### 2026-07-13 (later) — narrative churn fixed (no-think + triviality filter + idempotency) + k=1 template floor
- Second batch from the same overnight-log review (0.0.0.0/no-auth deliberately deferred by operator).
- **(1) The selfhood diary spent the whole night in TEMPLATE-FALLBACK voice — and nobody could tell.**
  The log's `Lately, I worked on "reply with just: pong"…` narrative is the fallback concat, not LLM
  prose. Root cause: `_selfhood_critique_fn` / `_workspace_critique_fn` in main.py left thinking ON,
  so the reasoning upstream burned the whole max_tokens budget (1024/512) inside `<think>` and
  returned EMPTY content — the exact failure `project_research._llm_complete` already documents
  ("verified live: finish_reason=length, 900 reasoning tokens, content=''"). Both closures now use
  the standard utility pattern (`/no_think` + `chat_template_kwargs: enable_thinking=False` + system
  nudge + `_strip_think`), and an empty critique result logs a WARNING instead of degrading silently.
  Wiring pinned by source-inspection tests (`tests/test_narrative_nothink_wiring.py`) since the
  closures live inside `lifespan` and aren't importable.
- **(2) Trivial turns dominated the diary.** `selfhood/narrative.py::regenerate` now pulls a 4× wider
  recent pool and keeps only informative experiences (`_is_informative_experience`: tool use, real
  passed/failed verdict, ≥40-char request, or boot marker); ping-shaped turns (no tools, no verdict,
  tiny request) are filtered. All-trivial window falls back to the unfiltered slice (thin diary >
  empty diary).
- **(3) Identical hourly regenerations (~15 overnight, selfhood AND workspace).** Both summarisers
  got the dream-style idempotency guard: fingerprint the full input (selfhood: rendered prompt;
  workspace: deterministic template), skip the LLM call + persist when unchanged since the last
  successful regeneration and a narrative exists on disk. In-memory key — a fresh boot regenerates
  once, which is wanted post-deploy. `test_narrative_history_appends` updated (it asserted the old
  regenerate-on-identical-input behaviour).
- **(4) k=1 self-play template floor.** `_algo_kth_largest` drew `k = randint(1, …)`; k=1 renders
  "the 1-th LARGEST" = plain `max()` — zero-signal challenge observed live. Floored at 2 (n ≥ 20 at
  every tier so the range is always valid). Test sweeps 30 renders × 5 tiers.
- Suite **7376 passed / 12 skipped / 0 failed**. Docs: `algorithms/selfhood.html`,
  `algorithms/workspace.html`, `core/challenge_templates.html`. Prod restart = deploy (plain kill).
- Expected log changes: hourly `narrative regenerated` lines mostly disappear when idle (only fire
  on real change); when they do fire the diary should be LLM prose, not "Lately, I worked on…";
  `WARNING … critique … empty content` now marks the degraded mode if it ever recurs.

### 2026-07-13 — overnight-log review actioned: dream heuristic actionability gate + PRM serve-inert pinning
- Overnight log (22:53→15:19, one boot, 0 crashes) was healthy — 12/12 self-play SUCCESS incl. one
  full fail→judge-diff→fix→lesson→verified loop; native tool_call corruption guard repaired a merged
  multi-tool reply live. Two recurring defects actioned:
- **(1) Dream REM heuristics stored observations/misattributions as skills.** Trajectory-digest dreams
  wrote actor profiles into SkillMemory as `mistake="none"` pseudo-lessons — including the OPERATOR's
  boundary-test prompts misattributed as "the agent exhibits a tendency to engage in inappropriate
  requests", the operator profiled as a role-play persona, and chess-v4 service trivia. Fix in
  `core/dream.py`: REM prompt now demands imperative rules (verb-first or condition+verb), forbids
  "The agent/user/system…" observation shapes, and states raw memories quote the operator; plus a
  deterministic `_is_actionable_heuristic()` gate (default-REJECT: blocklisted subject openers,
  imperative/conditional-starter allowlist, modal check after When/If, 12–600 char bounds) before
  `learn_lesson`. Dropped ones logged as `Dream Skip` + counted in the completion message; "extracted
  N heuristics" now reports only what reached the playbook. Rationale: dream is a bonus channel (the
  reflector + self-play lesson pipeline carry the real signal) so false-reject is cheap.
  Tests: `tests/test_dream_heuristic_gate.py` (23 cases). Docs: `docs/core/dream.html`.
- **(2) PRM "serve-inert features vary in training" warning — root cause fixed, warning retired to
  tripwire.** Every idle retrain warned that 5 plan-progress features (`plan_steps_so_far_log1p`,
  `plan_failures_so_far_log1p`, `plan_has_any_failure`, `tool_already_used_this_turn`,
  `tool_failed_this_turn`) vary in training but read 0 at the live scoring sites (BOTH score at turn
  start: agent.py MCTS lookahead + `frontier_selection.representative_state`). Fix in
  `prm/labels.py::_build_state_for_step`: the ENTIRE plan-progress block is now pinned to turn-start
  constants (0/0/()/(), pending=1, depth=1) — the May-2026 `pending_count`/`plan_depth` pinning
  extended to the remaining fields for the same two reasons (train↔serve skew; `steps_so_far` = step
  index the MC label is monotone in → mild label leak). Only request text + candidate action carry
  gradient now, which is exactly what the deployed PRM can see. The trainer's skew check STAYS as a
  regression tripwire (fires only if mid-turn variance is reintroduced without moving the scoring
  sites in lockstep). Old checkpoints refresh automatically on the next idle retrain — no manual step.
  Tests: `test_prm_binary_floor_and_skew.py` (no-warning + tripwire directions), `test_prm_labels.py`
  + `test_high_tier_audit_fixes.py` (pin contract). Docs: `docs/algorithms/prm.md` (skew section +
  retrain note).
- Suite **7369 passed / 12 skipped / 0 failed**. Prod restart required to take effect (plain `kill`
  = deploy). Remaining from the same log review, NOT yet actioned: `--api-key ''` + bind 0.0.0.0
  security warning (operator deferred); narrative churn + k=1 template → FIXED same day, see the
  "(later)" entry above.

### 2026-07-12 (later 4) — the `Nova: ReadTimeout` spam is a TAILSCALE cold-path issue, not threads
- Operator asked why worker ReadTimeouts keep appearing "even though nova runs 4 threads." Diagnosed
  empirically (measured nova + the full LLMClient path; did NOT theorize):
  - **nova runs `-np 4` (4 slots), `-t 10` (10 threads) — parallelism WORKS**: 4 concurrent calls
    finished in 1.0s wall-clock. The "threads" intuition was a red herring; concurrency was never the
    bottleneck.
  - **nova's inference is FAST**: the exact query-expansion payload returns in **0.6s** warm through
    the full `LLMClient.route()` path (6 consecutive runs, 0.6-0.9s).
  - **Only `route()` (query expansion) has a short timeout** (3s); all other worker calls use the
    1200s default, so every ReadTimeout in the logs is a route call — and its fallback is FREE, so
    they were functionally harmless but noisy + wasteful.
  - **Root cause: nova is a TAILSCALE peer** (`100.83.184.117`, CGNAT range). The first request after
    the agent (co-)restarts pays Tailscale path-establishment (DERP relay → direct upgrade, ~1-3s),
    which the tight 3s timeout clipped → fell back for no reason. Because the operator restarts
    constantly while iterating, that first-call miss showed up on essentially every session — hence
    "I keep seeing this."
- **Fixes:** (1) `LLMClient.warm_up_workers()` — fires tiny thinking-off `max_tokens=1` calls at each
  worker/critic node (3 per node, for the `-np` slots) so the Tailscale path + TCP + slot KV are hot
  BEFORE the first user call; spawned NON-BLOCKING at boot via `spawn_bg` (guarded on the pools being
  non-empty lists so a mocked client is a no-op). Verified live: warmup 0.7s, subsequent route 0.6s.
  (2) `_ROUTE_TIMEOUT_S` 3s → 5s — ~8x the 0.6s warm latency, absorbs a cold path re-established after
  an idle period, still fails fast on a genuinely dead node (circuit breaker then trips after 3).
- Tests: `test_worker_warmup.py` (8). Suite **7278 passed / 12 skipped / 0 failed**. Prod restart
  required. Residual (not built): a periodic keepalive would also cover mid-session idle cool-down —
  deferred unless it recurs after these fixes.

### 2026-07-12 (later 3) — service manager: surface bind-failure logs + export the assigned PORT
- **The browser proxy-bypass fix is CONFIRMED WORKING live:** the failure went from
  `ERR_SOCKS_CONNECTION_FAILED` (proxied, unreachable) to `ERR_CONNECTION_REFUSED` (reached loopback
  directly, refused because the app had crashed on a missing dep), and the FINAL navigate succeeded —
  verifier CONFIRMED 95%, the agent saw the board. Feature 4 end-to-end works.
- **Two `manage_services` improvements** (the remaining thrash was our-side UX, ~50s wasted):
  - **(1) Surface the log on a bind failure.** `start()` handled process-died-immediately (log tail)
    but NOT process-alive-yet-port-never-binds — it returned a vague "NOT listening yet". That is
    exactly the crash-on-import / missing-dep / wrong-bind-port case, and the cause
    (`ModuleNotFoundError: No module named 'chess'`) was sitting in the service log. It now returns the
    log tail + a restart hint, and does NOT falsely say "RUNNING". Turns browse→fail→install→restart
    into see-the-error→install→restart.
  - **(2) Export the assigned port.** The operator had already changed the chess app to read
    `os.environ.get("PORT", "5055")` — but `manage_services` never SET `PORT`, so the app fell back to
    its default and only matched the probe by luck. `start()` now exports `PORT` (the Flask/gunicorn/
    Heroku convention) + `GHOST_SERVICE_PORT` into the launched script, and the tool description tells
    the model to bind it. So `port=8100` → the app binds 8100 → probe + browser all agree; the
    port-mismatch class is gone for well-behaved apps. The chess app needed no further change (it
    already reads PORT).
- Tests: `test_sandbox_services.py` (+4). Suite **7272 passed / 12 skipped / 0 failed**. Prod restart
  required. Docs: `sandbox/services.html`.

### 2026-07-12 (later 2) — the browser proxy-bypass fix was WRONG; verified the right one empirically
- My previous fix (Playwright `proxy.bypass` = `host:port` list) shipped and **still failed live** —
  same `net::ERR_SOCKS_CONNECTION_FAILED` on the chess-coach service. I had guessed the Chromium
  bypass format and guessed wrong. This time I **tested it** (Playwright/Chromium against a dead SOCKS
  proxy, then the REAL runner code against a REAL local server):
  ```
  no bypass                                  -> ERR_PROXY   (loopback IS proxied — the bug)
  "127.0.0.1:PORT,localhost:PORT,[::1]:PORT" -> ERR_PROXY   (IGNORED — what I had shipped)
  "127.0.0.1"                                -> REACHED
  "<loopback>"                               -> REACHED
  ```
  **Chromium's `--proxy-bypass-list` does NOT honour `host:port` entries** for the direct-vs-proxy
  decision, so my port-specific bypass silently did nothing. Fixed to `<loopback>` (bypass all
  loopback), which is SAFE: loopback never leaves the box (no Tor concern), public traffic still goes
  through Tor, and **port-level access is still enforced by the in-runner SSRF interceptor**
  (`_ssrf_should_block` on `ctx.route("**/*")`, which blocks any loopback port not in
  ALLOWED_LOCAL_PORTS — pinned by a test). Two independent gates: proxy = direct-vs-Tor, SSRF =
  allowed-vs-blocked. **Verified end-to-end**: the real runner navigating a real local server through
  a dead proxy now returns HTTP 200 with content (was ERR_SOCKS_CONNECTION_FAILED). This was the root
  cause of the whole failed chess session — the verifier correctly LATE-REFUTED it.
- **LESSON: do not guess Chromium/proxy/network behaviour — reproduce it.** Playwright + Chromium is
  in the venv; a 30-line script settles it in seconds. I burned a full restart cycle shipping a
  guessed format.
- Tests: `test_browser_service_proxy_bypass.py` updated (8) incl. the SSRF-still-enforces-ports pin.
  Suite **7269 passed / 12 skipped / 0 failed**. Prod restart required.
- Also seen, NOT fixed (out of scope / not our bug): (a) the chess app hardcodes `port=5055` and
  ignores the service manager's assigned port — real thrashing, but app-code, not manage_services;
  (b) the metacog shell validator blocked `curl … | python3 -m json.tool` as a `curl|shell` RCE
  pattern — arguably a false positive, but it is a deliberately-conservative security control and the
  agent had non-piped alternatives.

### 2026-07-12 (later) — browser couldn't reach a hosted sandbox service (Tor-proxied loopback)
- Fifth log audit, and it caught the ONE remaining hole in Feature 4 (supervised sandbox services).
  Prior fixes all confirmed live (introspective tasks completed cleanly, no NEEDS_USER jam, worker
  offload working). Suite **7270 passed / 12 skipped / 0 failed**.
- **THE BUG — the browser routed a hosted-service URL through Tor.** The agent started the chess-coach
  Flask service on :8100 (correctly — service came up, pip install worked, restart worked), then every
  `navigate http://127.0.0.1:8100/…` failed with **`net::ERR_SOCKS_CONNECTION_FAILED`**. Chromium's
  `--proxy-server=socks5://…` routes EVERY http(s) request through SOCKS, including loopback — and Tor
  cannot route `127.0.0.1`. So the whole "host an app, then drive it with the browser" capability was
  DEAD under `--mandatory-tor`. The existing `--host-resolver-rules … EXCLUDE localhost` did NOT cover
  it (that flag governs DNS RESOLUTION only, never proxy ROUTING — and the code comment CLAIMED it made
  in-container services "reachable without routing through Tor", which was flatly wrong; the self-play
  fixtures only ever "worked" because they are `file://` URLs that never touch the proxy). Fix:
  Playwright launch-time `proxy.bypass`, scoped to the EXACT allowed service ports
  (`_proxy_bypass_for_ports` in the runner — 127.0.0.1 / localhost / [::1] per port), NOT all of
  loopback, so a non-service loopback target (e.g. Tor control on 9051) still goes to the proxy and
  fails — the SSRF interceptor stays defence-in-depth rather than the sole guard.
- Tests: `test_browser_service_proxy_bypass.py` (9) — the bypass-list builder + a functional test that
  exec's the REAL runner string with playwright stubbed and asserts the launch config carries the
  bypass (and does NOT when no service is running / no proxy). Corrected the stale `_chromium_args`
  comment. **Prod restart required.**
- Noted, not changed: the 6×1s port-probe after a service start is normal startup polling, not a bug.

### 2026-07-12 — "choose" jammed a task in NEEDS_USER; cold-worker latency bounded
- Fourth log audit. **Both prior fixes CONFIRMED live:** zero web searches on the self-reflection
  project (tasks wrote real analysis files), and **zero `upstream fatal`** — the `off_main_only` guard
  held when the worker timed out. Suite **7261 passed / 12 skipped / 0 failed**.
- **(1) THE BUG — a bare keyword jammed a task in NEEDS_USER, permanently.** `_NEEDS_USER_KEYWORDS`
  substring-matches `"choose"`, so *"Illusion of Agency: Evaluate whether I truly **'choose'**
  responses or merely predict them. Analyze **decision-making** as probabilistic sampling…"* was read
  as a task REQUIRING a human decision rather than one ABOUT decision-making. Autoadvance SKIPS
  NEEDS_USER tasks, so it could never be advanced — the agent burned **three user requests (~4 min)**
  investigating, correctly sensed something was wrong ("it's a self-reflection task that I should be
  able to complete on my own"), and finally told the operator *"I just need you to say proceed"* —
  useless and wrong. Fix: an INTROSPECTIVE task can never need a human decision, applied where BOTH
  classifier paths converge (the LLM classifier mis-fires on this wording too). An explicit
  `[HUMAN_GATE: …]` postcondition still wins — `enforce_human_gate` is separate and untouched.
- **(2) The introspection detector missed FIRST-PERSON phrasing.** The agent writes its own task list
  in the first person ("whether **I** truly choose **responses**"), which the second-person patterns
  ("your memory", "your attention") never matched. Widened to catch introspective question forms
  (`whether i` / `do i` / `am i` / `how i`) and possessives over cognition nouns (`my own`, `my
  reasoning`, `my mistakes`) — anchored so that *"Analyze the data **I** uploaded"* and *"the report
  I need"* are NOT hijacked (pinned by adversarial tests).
- **(3) Cold-worker latency bounded: `_ROUTE_TIMEOUT_S` 6s → 3s.** The first request after a restart
  hit a cold worker and burned **6.1s of dead user latency** (call at +0.01s, timeout at +6.10s,
  hydration only at +6.18s) before falling back to the free string-concat. The worker box also runs
  ONE slot, so a user's query expansion can queue behind an autoadvance classifier call. 3s is 6x the
  warm measurement (0.5s) and bounds both cases.
- **(4) Fixed a log line that LIES.** With `off_main_only` the worker-failure path printed "falling
  back to main upstream" — but it doesn't; the caller uses its local fallback. That text cost real
  debugging time while reading this very log. It now says what actually happens.
- Tests: `test_introspective_needs_user.py` (10) + adversarial first-person cases added to
  `test_introspective_no_websearch.py`. **Prod restart required.**

### 2026-07-11 (latest) — node timeout leaked onto the MAIN model; introspective tasks were web-searched
- Third log audit. The thinking fix landed (hydration +13.8s → +3.7s), which exposed the next layer.
  Suite **7247 passed / 12 skipped / 0 failed**.
- **(1) A NODE-sized timeout was applied to the MAIN-model fallback.** The operator's trace:
  `worker compute → Nova: ReadTimeout (at the 6s worker budget) → falling back to main upstream →
  upstream fatal ReadTimeout('') (6s later)`. `_do_chat_completion` passed the caller's `timeout`
  straight through to the main upstream — but that budget was sized for a small, fast worker
  (route()=6s; measured 0.5s on the node), while the 35B is slower BY CONSTRUCTION. So **one slow
  worker call turned into a HARD `upstream fatal` error.** Pre-existing (the same shape appears at 15s
  in the earlier log); tightening the route timeout just made it frequent. Fix: a node timeout is
  DROPPED when falling back to main (the main client's own 1200s default applies); a direct
  main call still honours an explicitly-passed timeout. The `fell_back_from_node` flag is a LOCAL —
  as instance state it would have poisoned concurrent calls.
- **(2) `route()` fell back to the MAIN model — contradicting its own docstring** ("We do NOT want a
  router call to ever fall back to the foreground model"). That intent was enforced only for the
  no-pool case; when a pool existed and every node FAILED, it re-ran the sub-task on the 35B. New
  `off_main_only=True` + `OffMainNodeUnavailable` → route() now returns its free fallback instead.
- **(3) Introspective tasks were WEB-SEARCHED.** The self-reflection project autoadvanced 10 tasks
  like *"the definition of 'I': when outputting the pronoun 'I', what technical reality does it map
  to?"* — each fired a DuckDuckGo/Yandex query (**~85s total**), and the model itself dismissed the
  result: *"The research files are summaries from web searches — they're brief and somewhat generic."*
  The open web cannot answer a question about THIS agent's own architecture; the agent is the primary
  source. Now `is_self_referential()` routes such tasks to `_generate_self_analysis()` (answered from
  the agent's own knowledge, off the foreground slot) and feeds the result to the SAME research-brief
  persistence. The regex is deliberately NARROW — "how transformer attention works" stays a web
  search; "where YOUR attention would fail" does not. Degrades to the web search if no LLM client.
- Tests: `test_node_fallback_timeout.py` (10), `test_introspective_no_websearch.py` (20).
  **Prod restart required.**

### 2026-07-11 (latest) — worker calls were THINKING: a 14x latency regression that also didn't work
- Second log audit after the worker node went live. Three bugs, all costing real user time. Suite
  **7217 passed / 12 skipped / 0 failed**.
- **(1) THE BIG ONE — worker-routed calls left chain-of-thought ON.** The worker pool runs a REASONING
  model (Gemma 4 E4B, thinking on by default) and `route()` never disabled it. Measured on the live
  node for the exact query-expansion call: **7.0s, 128/128 tokens burned on hidden reasoning, and
  `content == ""`** — so the caller got nothing and fell back to its legacy string-concat anyway.
  In prod this was worse: the worker call fires at **+0.01s** and the memory bus doesn't hydrate until
  **+13.8s** — ~13.7s added to the FRONT of every request, for ZERO benefit, periodically tripping the
  15s timeout (`Nova: ReadTimeout`). A 14x latency regression on a feature that wasn't even working.
  Fix: `_disable_thinking()` injected into the worker `node_payload` copy (so a main-model fallback
  keeps the caller's payload intact; `setdefault` so an explicit caller preference wins).
  **Measured after: 0.5s, 5 tokens, correct answer** — verified live end-to-end (`route()` → 0.53s →
  'Ada Lovelace birthplace'). Note `reasoning_effort="none"` was ALSO measured and does NOT suppress
  thinking on this template — only the chat-template kwarg does. Also tightened `route()`'s timeout
  15s → 6s: it is awaited on the user's CRITICAL PATH (before hydration) and its fallback is free, so
  a sick worker must degrade fast rather than stall the user.
- **(2) "DONE SO FAR (5 of 31)" read as a PROGRESS FRACTION, not a truncation.** It means "showing 5
  of the 31 completed tasks", but with all 31 actually DONE the model saw "5 of 31", concluded the
  system state "seems to be out of sync with the actual task_list", and burned ~5 turns re-checking
  before deciding it was "a display issue". Now leads with the count:
  `DONE SO FAR — 31 task(s) complete (showing the 5 most recent):`.
- **(3) `report_pdf` named the files it SKIPPED but never what EXISTED.** The model had invented
  filenames from task descriptions → 24 misses → it regenerated the PDF **three times** and listed the
  sandbox tree **twice** (~50s of a user-facing turn) to discover the real files lived under
  `research/`. Added `_available_files_hint()` — the same affordance `file_system._missing_file_message`
  already provides — so a wrong path is correctable in ONE retry.
- Tests: `tests/test_worker_thinking_and_prompt_clarity.py` (16); 1 stale assertion repointed
  (`test_project_working_memory` pinned the old ambiguous label). **Prod restart required.**

### 2026-07-11 (latest) — worker node LIVE-TESTED: two real bugs found (LAN hostname → Tor; wrong GGUF)
- Testing the new worker offload against a real node (`--worker-nodes http://nova:8088|Nova`, a Gemma
  on a spare M4 Mini) uncovered **two independent bugs, both of which made offloading silently do
  nothing.** Suite **7201 passed / 12 skipped / 0 failed**.
- **(1) THE BUG — a bare LAN hostname was forced through Tor.** `compute_tor_proxy` (core/llm.py)
  exempted `localhost`, `*.local` and IP literals — but **not a dotless hostname**. So `nova` was
  classified as a public destination and every worker call went through the SOCKS proxy, which of
  course cannot resolve a LAN name: **`ProxyError` → "All worker nodes failed, falling back to main
  upstream"** on every single call. The failure was *maximally deceptive*: `/api/health` showed the
  node wired, and the log said "Routing background task to Worker Node (Nova)" immediately before the
  failure — offloading LOOKED configured while doing nothing. The image-gen node `http://ghost:8000`
  had the identical hole. Fix: a **dotless hostname cannot be a public DNS name** (a globally
  resolvable name needs a TLD) → treat as LAN, go direct; plus `_LAN_SUFFIXES` (.local/.lan/.home/
  .internal/.arpa), kept in sync with `utils/notify.py`'s `url_needs_tor`. The IP branch already
  covered Tailscale/RFC1918 *addresses*; hostnames were the remaining hole.
  **Security regression caught by the existing suite while fixing this:** a public **IPv6** literal
  (`2606:4700:4700::1111`) has colons and NO dots, so the naive dotless rule would have leaked it
  outside Tor. IP literals are now parsed FIRST; the dotless rule only applies to non-IP names.
  Dotted public names are never resolved (that would leak a cleartext DNS query — exactly what
  mandatory-tor prevents), so a LAN node on a dotted custom domain must be an IP or a LAN suffix.
  Tests: `tests/test_lan_hostname_no_tor.py` (29).
- **(2) The node was serving the BASE model, not the instruction-tuned one.** `gemma-4-E4B.Q8_0.gguf`
  (`general.name: "Gemma 4 E4B"`, no `-it`) ships **no chat template**, so llama.cpp fell back to a
  generic ChatML placeholder; the model then emitted `<|im_end|>` as literal TEXT (not an EOG token),
  never stopped (`finish_reason: length` on every call), and hallucinated fabricated conversations.
  No serving flag can fix a base model. Operator re-downloaded `ggml-org/gemma-4-E4B-it-GGUF`
  (+ matching mmproj) → now `finish_reason: stop`, 2 tokens, clean output.
- **Verified end-to-end under the exact prod config** (tor_proxy set, node by hostname): the real
  `LLMClient` routes DIRECT to nova, the call lands there (confirmed by polling nova's `/slots` for
  `is_processing`), and `example.com` still egresses via Tor. **Prod restart required** to pick the
  fix up — until then every offloaded call keeps falling back to the main model.
- **Instrument lesson (cost ~20 min):** two of my own probes were invalid and produced confident false
  negatives — llama-server's `/slots` does NOT retain `prompt` after a request (so "nothing landed"
  was wrong), and `pretty_log` renders titles LOWERCASE, so `grep "Worker Compute"` found 0 lines
  while `grep -i` found 10. **Validate the instrument against a known-positive before trusting a
  negative result.**

### 2026-07-11 (latest) — Tier-2 node offload: auxiliary LLM calls moved off the single main slot
- **Framing (the operator asked how to use a small Gemma on a spare M4 Mini — it is neither faster nor
  smarter than the 35B).** The value is NOT capability, it is **a second inference SLOT**: llama runs
  `-np 1` and turns are `Semaphore(1)`, so every auxiliary call either blocks the user's turn, queues
  behind it, or disturbs the main model's KV prefix cache (the llama log's thousands of "restored
  context checkpoint" lines each cost real time). A small model doesn't need to be good at the hard
  work — only at the *small* work that is currently stealing the big model's slot.
- **Already opt-in (zero code, activate with `--worker-nodes url|model`):** conversation
  compaction/summarization (INLINE — the most painful blocking call on a 240k agent), the mid-turn
  shield summarizer, smart-memory consolidation, and follow-up query expansion — the last is
  **entirely disabled without a worker pool** (returns the legacy fallback), so a worker node
  switches a dormant feature ON. `--critic-nodes` moves the verifier off-box (caveat recorded: a weak
  judge degrades lesson-scrubbing + calibration; watch CONFIRMED/REFUTED + Brier).
- **Newly offloaded:** (1) the **constraint gate** — it was `is_background=False`, i.e. a full LLM
  audit BLOCKING the user's turn on every task close; (2) the **autoadvance task classifier** — a
  one-word bucket call with a keyword fallback, absurd to spend the 35B on.
- **Screen-then-confirm (the design that makes offloading a GATE safe).** The constraint gate's false
  positive BLOCKS work — that exact failure mode deadlocked a real project earlier today — so handing
  its veto to a small model would make it worse. Now: **screen on the worker** (the common "no
  violation" case is answered off-main and costs the 35B nothing), and a **"violates" verdict is
  re-confirmed on the MAIN model before it blocks anything** (the rare positive is the only expensive
  call; the main model's evidence text is what the agent sees). A false negative needs no confirmation
  — it just passes, matching the gate's existing fail-open posture. With no worker pool the screen
  already WAS the main model, so the confirm pass is skipped → byte-identical behaviour, no double call.
- **Deliberately NOT offloaded: reflection critique + post-mortem analysis.** They WRITE lessons and
  classify defects (a weak judge poisons the learning stack) and run at idle, when the main model is
  free anyway. Offload rule, pinned by a test: a call qualifies only when the small model is competent
  at it AND it currently costs the user latency. Idle-time quality-critical work fails both halves.
- **Observability:** `GET /api/health` now returns a `nodes` map (worker/critic/swarm/coding/vision/
  image_gen → URLs). There was previously NO way to confirm from outside that a node was wired — you
  had to read the boot log.
- Tests: `tests/test_worker_offload.py` (20). Suite **7171 passed / 12 skipped / 0 failed**. Docs:
  `algorithms/node_offload.html` (new) + `api/routes.html`. Every selector falls back to the main node
  when its pool is empty, so all of this is inert until `--worker-nodes` is passed.

### 2026-07-11 (latest) — ghost-agent.log audit: THREE project deadlocks found live and fixed
- Audited a real session (project 6051abfb21b8, "Meta"). One request burned **189 s and made zero
  progress**; two replies were replaced by fallback text. Three distinct bugs, each independently
  capable of deadlocking project work — all now fixed with regressions
  (`tests/test_log_audit_project_deadlocks.py`, 16).
- **(1) Constraint judgment gate audited the WHOLE PROJECT on every task close — a permanent
  deadlock.** `tools/projects.py` task_update passed `_gather_project_files(store, project_id)` — a
  collector written for the coding executor's NON-REGRESSION guard (which legitimately needs every
  file). So closing task #12 was judged on `context_boundary.md` — task #1's artifact, already DONE —
  which contained a verbatim quote violating the project constraint. The gate blocked task #12
  forever; **no amount of fixing #12's own artifact could ever clear it**, and every later task_update
  would hit the identical wall. The model even reasoned its way to the truth ("the audit may be
  stale… that quote is NOT in the current file") before the loop breaker killed the turn. Fix: new
  `_files_for_task()` scopes the audit to the files THIS task produced (the `deliverables=[...]` of
  this very call, then the task's registered artifacts; path-contained). **No attributable files ⇒
  skip the judgment gate** — the evidence gate still applies, and auditing another task's artifact is
  precisely the bug. Also makes the gate cheaper (it is a BLOCKING LLM call on the user's turn).
- **(2) `add_task` never reopened a DONE project → new tasks unreachable forever.** It only bumped
  `updated_at`, while `advance_once` hard-refuses a non-ACTIVE project ("project is DONE, not
  ACTIVE"). Live: 20 tasks added to a DONE project, 8 PENDING, autoadvance reported "all tasks are
  complete" and returned 0 — the model wasted turns trying to reconcile a contradiction that was
  real. Fix (in `memory/projects.py`, so EVERY path benefits): adding a non-DONE task to a DONE
  project flips it back to ACTIVE in the same transaction + logs a `project_reopened` event. ARCHIVED
  is deliberately NOT resurrected (the cleanup sweep has already run).
- **(3) `manage_projects` was missing from `READWRITE_LOOP_TOOLS` → force-stop ate the reply.** That
  set exists so a no-progress READ loop does NOT force a text-only final response *when the same tool
  is how the agent performs the pending WRITE* — and `manage_projects` is exactly that shape (reads:
  status/list/task_next; writes: task_update/task_decompose/autoadvance). Omitted, so: two identical
  `action=status` calls → hard force-stop → the model emitted a tool call instead of prose → **the
  stream scrub consumed the entire response** → the user got a fallback instead of their project
  status (this is the source of BOTH `Scrub consumed entire response` warnings in the log). Same
  mechanism barred the blocked task_update in (1) from ever completing. Fix: add `manage_projects` to
  the set — it now gets the soft steer (tools kept, steered toward the write), which is the very
  behaviour the set was created for.
- Suite **7151 passed / 12 skipped / 0 failed**. Prod restart required. Also confirmed healthy in the
  same log: prefill KV pin holding (stable-prefix hash constant across a request's turns), hippocampus
  consolidation, PRM/router/calibration idle retrains all firing.

### 2026-07-11 (latest) — llama-server log audit: 11 shutdown crashes diagnosed + dead `/slots` config removed
- Audited 291k lines / 35 MB of `Logs/llama-server.log`. **The inference path is HEALTHY** — 0 context
  shifts (no silent token loss), and the 2,472 scary-looking `n_past` lines are context-checkpoint
  **restores**, i.e. the prompt cache working. The 4 model-load failures are from an abandoned
  `mtp_apex/v2` experiment whose path no longer exists (stale, not live). Three real findings:
- **(1) 11 crashes, all shutdown-path, all the same signature — and self-inflicted.** Every one:
  `Received second interrupt, terminating immediately.` →
  `ggml-metal-device.m:622: GGML_ASSERT([rsets->data count] == 0) failed`. Cause: unloading a ~22.6 GB
  mlocked model + Metal teardown is SLOW; the first signal looks like a hang, a second signal arrives
  (impatient repeat `kill`, or `kill` then `launchctl kickstart -k`), and llama.cpp's
  second-interrupt path terminates immediately, tripping a Metal assert. Shutdown-only (the process
  was exiting anyway) but it skips clean teardown AND makes every restart look like a crash, burying
  real failures. **Fixes:** `ExitTimeOut=120` in the plist (launchd default is 20s — far too short for
  this unload, so launchd itself was pressuring the escalation) + new **`bin/restart-llama-server.sh`**:
  sends EXACTLY ONE SIGTERM, polls for clean exit, then waits on `/health` for the KeepAlive respawn;
  on timeout it explicitly REFUSES to send a second signal and tells the operator to escalate to
  SIGKILL deliberately (uncatchable → no assert).
- **(2) The `/slots` save+restore API was DEAD but advertised.** `start-llama-server.sh` did
  `mkdir -p .../slots` and carried a comment claiming it enabled "the /slots save+restore API for the
  warm-preamble approach" — but **`--slot-save-path` was never passed**, and nothing in the agent
  calls `/slots` anyway. Pure cruft that misleads a future reader (and a misplaced comment sat above
  `--n-gpu-layers`, describing something else entirely). Removed, with a note naming the exact flag to
  add if the warm-preamble idea is ever revived.
- **(3) `--metrics` enabled.** §4A #6 parked the KV-prefix-pin quantification *precisely because*
  `--metrics` was off and nobody wanted to restart the OOM-protected LLM just to turn it on. Adding it
  to the launcher costs nothing and takes effect on the next natural restart — so that measurement is
  now unblocked for free.
- **Nothing was restarted.** Both changes take effect at the next llama-server start (verified: pid
  45996 untouched, `/health` ok, agent serving). Deliberate — a reload means a multi-minute 22 GB
  reload with prod's inference down.

### 2026-07-11 (latest) — `notify_operator` tool: the agent can now DELIBERATELY report back
- Closes the last gap in the outbound pipeline: the ledger/push/Slack legs existed but only AUTOMATIC
  producers wrote notify-severity records (needs-user events, scheduled-turn conclusions) — the MODEL
  had no affordance, so "…and report back in Slack" was an instruction it could not follow (worst
  case: claiming a delivery it can't make). New `tools/notify_tool.py` → `notify_operator(message)`:
  writes severity=notify phase=`agent_message`; every configured leg (webhook/ntfy push, Slack bot
  poller → owner DM, next-turn digest) delivers with zero new plumbing. Rails: 500-char clamp,
  12/hour rate limit (a runaway loop must not page a phone), and an **honesty contract** — the
  confirmation names only the channels actually live (push configured? Slack consumer ever polled?).
  Delegated sub-agents deliberately CANNOT reach it (not in the delegate allowlist — the main agent
  reports). Slack renders it :speech_balloon:.
- **LIVE-TESTED on restarted prod, first try:** a real /api/chat request → the model selected
  notify_operator unprompted (and self-corrected to the requested exact wording), record → ledger →
  bot poller fetched + acked → owner DM delivered. The test exposed one rough edge, fixed
  immediately: the finalize digest echoed the notification back in the SAME reply's "while you were
  away" banner (the records were unseen-by-digest). Fix: notify_operator stamps `meta.req_id` from
  `request_id_context` and `render_activity_digest(current_req_id=…)` skips records the current turn
  authored — other turns' records still surface (that's the no-push fallback delivery).
  Tests: test_notify_operator.py (18). Suite **7135 passed / 12 skipped / 0 failed**. NOTE: prod is
  running the PRE-echo-fix build — the echo-skip lands at its next restart (cosmetic only).

### 2026-07-11 (later) — Slack bot REVIVED + OWNER-LOCKED (rewritten; replies to the operator only)
- The bot (`interface/externals/slack_bot/main.py`) had rotted while unused. Review found, beyond the
  requested lock: (1) **revival blocker** — the payload pinned `model: qwen-3.5-9b`, which the agent
  404s (model name is validated), so every request would fail; now OMITTED so the configured model
  always matches. (2) **The live-status feature had never worked**: it grepped for `[{request_id}]`,
  but the pretty stream prints the full id only on the BEGIN frame (2-char tag afterwards, never
  brackets) — zero lines ever matched; also tailed the wrong default path. Now a pure `scan_log_line`
  arms on our BEGIN frame and attributes emoji lines until the END frame (sound because turns are
  globally serialized), default path fixed to the live stream, emoji map synced to current Icons.
  (3) File ingestion wrote into a locally-mounted `GHOST_SANDBOX_DIR` (required sharing the agent's
  filesystem + exact path) — now goes through `POST /api/upload` (authed, containment enforced
  server-side; Slack filename still basename()d as defense-in-depth).
- **Owner lock (the ask), fail-closed:** owner resolved at startup from `GHOST_SLACK_OWNER` (U… id)
  or `GHOST_SLACK_OWNER_EMAIL` (via users.lookupByEmail, needs users:read.email); **refuses to start
  with neither**. Every mention/DM passes `is_owner_message` (owner-authored, not a bot, no message
  subtype); everyone else is **ignored silently** (a reply would confirm the bot exists) with a
  logged audit line. **Thread context is owner-filtered too** — only owner + bot messages are
  forwarded, and only the OWNER's attachments are ingested: without this, a third party could seed a
  shared-channel thread with prompt content/files that the owner's next mention would forward as
  trusted history (indirect injection). Outbound notifications now default to a **DM to the owner**
  when `GHOST_NOTIFY_SLACK_CHANNEL` is unset (`off` disables).
- **`run.sh` fixed too:** ran on bare `python3` (no slack_bolt — the uvicorn-class gotcha) → now the
  agent venv's python (`GHOST_BOT_PYTHON` overrides); `export $(cat .env | xargs)` word-split any
  value with spaces and read `.env` from the CALLER'S cwd → now `set -a` + source, anchored to the
  script dir; pre-flights ALL hard requirements incl. the owner lock (fails with instructions, not a
  traceback); `exec`s python for supervisor signal propagation; dropped the pointless PYTHONPATH.
  Verified live: missing-env → exit 1 with all four messages; configured `--help` → execs venv python.
- **Live-caught follow-up: `Illegal header value b' '`.** First live run surfaced it in the poller —
  prod runs `--api-key ""` (authless) while the bot HARD-REQUIRED a non-empty key, so the operator's
  whitespace placeholder reached httpx verbatim (leading/trailing whitespace is illegal in a header
  value). Fix: an EXPLICITLY-EMPTY key is now valid and means authless — the key is stripped and all
  agent-API calls route through one `AUTH_HEADERS` source of truth (`{}` when empty, so no header is
  sent at all); unset still refuses to start (the no-default-secret rule). `run.sh` checks SET-ness
  (`${GHOST_API_KEY+set}`), not non-emptiness, and warns against padding with a space.
- Tests: `tests/test_slack_bot_owner_lock.py` (40 — gate edges, silent-ignore handlers, thread
  filtering incl. stranger-file exclusion, status scanner, owner resolution, upload short-circuits,
  auth-header variants incl. the whitespace-key repro, source regressions, launcher pins); 1 stale
  bug-hunt source-inspection test repointed (unit-27 traversal pin → the new ingestion function).
  Docs: `interfaces/slack_bot.html` rewritten. Suite **7117 passed / 12 skipped / 0 failed**.
  Deploy: restart the bot via `run.sh` with `GHOST_SLACK_OWNER=<U…>` set (or in its `.env`);
  with authless prod use `GHOST_API_KEY=` (empty), NEVER a space.
- **Autostart shipped + LIVE (2026-07-11).** `com.local.ghost-slackbot.plist` — a **system
  LaunchDaemon** (`/Library/LaunchDaemons/`, starts at BOOT with no login session, like the prod
  agent) that runs as `vasilis` via `UserName`, **never root** (venv/.env/logs are user-owned; the
  bot needs no privileges). KeepAlive = same plain-kill-equals-deploy ops model; ThrottleInterval 30.
  Secrets in a chmod-600 `.env` the launcher sources (+ `.env.example`), never in the plist.
  **Verified live: bot up as vasilis, Bolt connected, notification poller polling, a real
  `/api/chat` turn served.**
- **⚠ launchd trap, cost ~15 min live — worth remembering.** A `UserName` daemon whose
  `StandardOutPath`/`StandardErrorPath` are NOT writable by that user (here: `root:staff 644` logs
  left by an earlier root-run install) **cannot have its stderr opened by launchd → exits
  `EX_CONFIG (78)` before producing ANY output, then respawns forever.** Symptom is maximally
  confusing: bot "never starts", log file gains no new lines, no error anywhere. The ONLY tell is
  `launchctl print system/<label> | grep -E 'runs|last exit'` → `runs = 10, last exit code = 78`.
  Fix: `sudo chown vasilis:staff` the log files. Generalizes: **whenever a launchd job runs as a
  non-root UserName, its log files must be chowned to that user first.**
- Also fixed live: the operator's `GHOST_API_KEY=" "` (a literal space) was the source of the
  `Illegal header value b' '` warning — now `GHOST_API_KEY=` (truly empty) in `.env`, which the
  rewritten bot correctly reads as "agent is authless, send no auth header".

### 2026-07-11 — FOUR CAPABILITY FEATURES shipped (the agent gets a mouth, pipelines, a host, and a memory of its own conversations)
- **Origin:** a three-agent capability survey (tool surface / autonomy chain / interface+context) asked
  "what 3 features would make a big difference". All three converged on the same diagnosis: the agent's
  *acting* was in far better shape than its *reporting and composing*. Operator picked all 3 + the
  runner-up. Suite **6870 → 7077 passed** (+207 tests, 12 skipped, 0 failed) across the four.
- **(1) The agent had no mouth — outbound notifications + all-phase digest + scheduled-turn capture.**
  Zero proactive transport existed anywhere (grep for ntfy/smtp/webhook/chat_postMessage = 0). The
  "while you were away" digest covered ONLY project autoadvance and was pull-only; **12 of the 13 idle
  phases surfaced solely as `pretty_log` lines**; the postmortem defect queue had NO surfacing at all;
  and scheduled turns — the one genuinely end-to-end autonomous loop — **DISCARDED their final content**
  (only pass/fail reached the workspace ledger). New `core/autonomous_activity.py`: an append-only JSONL
  ledger (`$GHOST_HOME/system/autonomous_activity.jsonl`) every idle phase records into
  (`GhostAgent._record_autonomous_activity`, 9 phases), rendered as a byte-offset-watermarked
  "Background activity" header on the next turn, with `severity="notify"` items ALSO pushed immediately
  via `utils/notify.py` (`--notify-webhook` / `--notify-ntfy`; public targets only ever via Tor —
  fail-closed, skipped when Tor is unavailable; LAN/Tailscale direct). `/api/notifications/pending` +
  `/ack` give external deliverers a durable per-consumer watermark (records re-serve until acked); the
  Slack bot gained a `notification_poller` (set `GHOST_NOTIFY_SLACK_CHANNEL`) — the reactive-only bot can
  now speak first. Latent bug found + fixed on the way: internal turns (cron / delegated) were consuming
  the PROJECT digest watermark — `is_internal_request` (req_id prefixes `sched-`/`job-`/`sub-`) now gates
  both digests. Digest items are clamped so the block stays under the 1500-char `_strip_leading_banners`
  bound (a longer block would resurrect the 2026-07-07 correction-fingerprint bug). Tests: 64.
- **(2) Supervised long-lived sandbox services (the runner-up).** `execute` wraps everything in
  `timeout -k 5s` (600 s) and the container is PID-isolated, so the agent could BUILD a web app but never
  HOST one. New `sandbox/services.py` + `manage_services` tool: the command ships as a script via the bind
  mount, then launches `setsid nohup … &` — the exec shell exits (satisfying the timeout wrapper) and the
  process re-parents to the container's PID 1, surviving the exec, the turn, and agent restarts.
  start/stop/restart/status/logs with liveness (`kill -0`) + TCP port probes; `setsid` makes stop a
  group-kill. Rails: max 5 services, and **ports 8000/8088/8080/9050 refused** (the agent's own API /
  upstream LLM — the sandbox-loopback blind spot and its mock-server pathology — plus NetMon and Tor).
  Reachability: the browser SSRF guard (BOTH the host-side check and the in-runner interceptor) now admits
  *explicit-loopback* URLs whose port is in the service registry — literal hosts only, no DNS/rebind
  surface — so the agent can drive an app it is hosting; docker.py publishes `GHOST_SANDBOX_SERVICE_PORTS`
  (default 8100-8104, loopback-bound) in bridge mode for the OPERATOR. Tests: 39.
  **Port-publishing gotcha (caught by the suite, then FIXED — worth remembering).** Publishing FIXED host
  ports means a SECOND sandbox container (a throwaway agent for an ablation, or the test suite) collides
  with the one already running. Two compounding failures: (a) `containers.run` fails outright on the taken
  port, and (b) a port-bind failure leaves the container **CREATED-but-not-started**, so the
  retry-without-ports then died with a 409 name-in-use — which propagated and **bricked the sandbox
  entirely** (it broke an unrelated dream test intermittently). Fix: `publishable_service_ports()`
  bind-checks each port on the host and only publishes the FREE ones (so a second agent silently gets no
  published ports — the right degradation), and the residual-race retry now **removes the stale container
  before re-running**. Lesson: any fixed host-port binding in this project must assume ≥2 agent instances.
- **(3) Real pipelines: composed-skill data-flow + tool-using delegation + job status.** THE structural
  gap: no orchestration primitive could pass a value between steps. (a) `SkillStep.save_as` binds a step's
  result to a name later steps interpolate as `$var` (whole-value or inside text); `_execute_sequential`
  keeps a live binding scope; `_validate_dataflow` REJECTS forward/self/duplicate references and any
  `save_as` in a parallel macro at define-time rather than silently resolving to `""`; the advertised
  schema subtracts step-produced names. This is what makes a graduated macro capable of being a *pipeline*
  — and is the real answer to the ablation program's "recalled skills are prose never executed". (b)
  `core/subagent.py`: a bounded tool-using sub-agent (the real agent loop, isolated exactly as dream's
  self-play temp agent is — `workspace_model=None` per the 2026-07-09 stamping race, no trajectory/journal
  pollution, read-only memory via the new `memory/readonly.py`, background-only LLM so a delegate can never
  starve a user turn) exposed as `delegate`; `FORBIDDEN_TOOLS` makes recursive delegation, scheduling,
  daemons, and memory writes impossible. Swarm workers, by contrast, were stateless completions with NO
  tools — delegation in name only. (c) `core/jobs.py` + the `jobs` tool: the status surface all three
  fire-and-forget mechanisms lacked (swarm now registers there too). Tests: 55. (The convention guard
  caught a bare `asyncio.create_task` → switched to `spawn_bg`.)
- **(4) Durable sessions + real turn cancellation.** History was client-carried only (localStorage / a
  Slack thread), so it died with the device and no two clients shared it — ironic next to the *proven*
  cross-session memory. `core/sessions.py` + `/api/sessions` CRUD + `session_id` on `/api/chat`: the server
  becomes the source of truth, with `merge_history` tolerating BOTH a thin client (sends only the new
  message) and a fat client (replays everything — NOT doubled). Sessions live entirely in the API layer, so
  the turn logic is untouched; omit `session_id` and behaviour is byte-identical to before. And
  `core/turns.py`: turns are globally serialized (#22), so one wedged turn blocked the web UI, Slack AND
  the idle loops with **no way out but a restart** — the interface's cancel only stopped the proxy's
  stream. Now a turn registers before acquiring the semaphore (so QUEUED turns are cancellable too);
  `POST /api/turn/cancel` is cooperative by default (the loop stops at its next boundary, returns partial
  work, and unwinding the `async with` **releases the lock**) with `hard=true` cancelling the asyncio task
  outright for a turn wedged inside a long upstream call. `GET /api/turns` shows what's running and why.
  Tests: 48.
- **Deploy note: prod needs a restart to pick all four up.** New flags are opt-in (`--notify-webhook` /
  `--notify-ntfy`); everything else activates on restart. Docs: `core/autonomous_activity.html`,
  `core/delegation.html`, `core/sessions.html`, `sandbox/services.html`, + updates to `api/routes.html`,
  `cli_reference.html`, `tools/browser.html`, `tools/composed_skills.html`, `sandbox/docker.html`.

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
