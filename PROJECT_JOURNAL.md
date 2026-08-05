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

**Voice loop — LOCAL since 2026-08-02 (needs an interface restart to serve).** `/api/stt` and
`/api/tts` no longer proxy anywhere: STT = `ffmpeg` transcode + **nova's Gemma 4 audio node**
(`-mm …mmproj-BF16.gguf`), TTS = macOS `say`. The old `PI_VOICE_URL` target is GONE
(`raspberrypi.local` doesn't resolve; `disorder:8000` listens on nothing) and that export in
`start-ghost-client.sh` is now inert. Hold the mic (input area, restored 08-02) → spoken reply for that
turn; typing turns speech off. Long-form recordings ingest via `knowledge_base(action='ingest_document',
filename='talk.mp4')` → ~12-min windows, timestamp-stamped chunks. Config (all defaulted):
`GHOST_AUDIO_NODE_URL` (TAILNET ip — a dotless/mDNS name is what stranded the last backend),
`GHOST_STT_MAX_SECONDS`, `GHOST_STT_MAX_TOKENS`, `GHOST_TTS_VOICE`, `GHOST_AUDIO_WINDOW_S`,
`GHOST_AUDIO_MAX_S`, `GHOST_AUDIO_MAX_TOKENS`. **Gotcha 1:** too small a `max_tokens` on any Gemma 4 audio
call returns EMPTY content with `finish_reason="length"` (thinking tokens are stripped by its template)
— both call sites raise on that shape rather than reporting silence. **Gotcha 2 — launchd PATH:** a
daemon's PATH is `/usr/bin:/bin:/usr/sbin:/sbin` and EXCLUDES Homebrew, so `ffmpeg`/`ffprobe`
(`/opt/homebrew/bin`) are invisible to a bare `which` while `say` (`/usr/bin`) resolves — this 503'd
every STT request in the deployed process while working from a shell. Both the interface and
`memory/audio_ingest.py` now use `resolve_binary()` (`GHOST_FFMPEG_BIN`/`GHOST_FFPROBE_BIN`/
`GHOST_SAY_BIN` → PATH → known prefixes). **Anything a daemon shells out to must not trust PATH.**
See `docs/interfaces/voice_server.html`.

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
| Harness-dimension failure attribution (lessons + work_logs) | `GHOST_FAILURE_DIM` env kill; `learn_lesson` chokepoint + finalize work_log (2026-07-19) | live |
| Failure-cluster distillation + project dream pass (REM pre-gate) | `GHOST_FAILURE_DISTILL` / `GHOST_FAILURE_ADJUDICATE` / `GHOST_FAILURE_DISTILL_MAX` env kills; `dream()` pre-gate (2026-07-19) | ⚠ **was NEVER live** — the mock-guard was always False under the production import shape (§4J, fixed 2026-08-04). Distillation additionally has a structurally unreachable cluster gate; project dream pass is now genuinely armed. ⚠ Distillation mints lessons at utility ≈0.77 against a live prune cutoff of 1.1183 — harmless while the prune is off, but the two must be reconciled before it is re-enabled. The manifest-backfill half now excludes RELEASED projects (it was breaking their immutability while logging success), reads a bounded 4 KB, and skips binaries. |
| Outcome-gated lesson utility (failure arm → `compute_lesson_utility` → prune) | `GHOST_LESSON_OUTCOME_UTILITY` env kill (`=0` → record-only); `_record_lesson_outcomes` both finalize paths + late-verdict drain (2026-07-24) | recording live and verified against data (422 succeeded / 151 failed retrievals across 35 of 50 lessons); the PRUNE half is **OFF** — see below |
| Low-utility lesson PRUNE (destructive, unattended, deletes vector twins) | `GHOST_SKILL_PRUNE=1` to enable; `SkillMemory.prune_low_utility`, called from the REM cycle | **OFF by default (2026-08-04)** — armed by the mock-guard fix, it then destroyed 13 lessons in two runs (one at 277 retrievals / 70% success) with no archive. Re-enable criteria: (a) the cutoff must stop being a *relative* bottom quartile that always finds victims, and (b) failure-distillation must stop minting lessons at ≈0.77 below a live cutoff of 1.1183, which guarantees its own output is deleted. Archive now fails CLOSED; quarantined rows are exempt. |
| Journal-mined self-play challenges | `journal_challenges.pick_stashed_challenge`; `journal_prob=0.75` when the frontier saturates | live, with a destructive-content denylist since 2026-08-04 (`_is_unsafe_challenge`). Mined challenges are real user turns replayed VERBATIM against the real toolset — a live stash record instructing `DROP TABLE` via `postgres_admin` was one pick from execution. |

**Re-enable criteria (why each OFF toggle is parked, not deleted):**
- `_MCTS_TURNSTART_ENABLED` — only with an **execution-grounded** value fn (not self-prediction).
  Its intended grounded replacement (verifier-judged best-of-N that SUBSTITUTES the winner) landed
  2026-07-07 as the async-critic bounded repair (§5 #18).
- `_SELFHOOD_PREFIX_ENABLED` — the prefix injects no facts/tools/constraints (pure token cost). The
  load-bearing selfhood path is the cross-session memory substrate (Track B), which IS proven.
- `_METACOG_ARBITER_ENABLED` — net-negative as built; superseded by #18.

**Kill criteria for the 2026-07-19 ON toggles** (spot-audit after ~2 weeks): kill
`GHOST_FAILURE_DIM` if the dimension distribution in work_logs reads as noise in spot audits
(misattribution > ~30% — the tag is a prior, not a verdict; the debug log carries the matched
signal for auditing). Kill `GHOST_FAILURE_DISTILL` if distilled lessons crowd playbook slots or
their `helpful_retrievals` stay at zero while occupying retrieval budget. Kill
`GHOST_LESSON_OUTCOME_UTILITY` (→ record-only) if the prune drops lessons whose `failed_retrievals`
came from failures they demonstrably did not cause (co-occurrence ≠ causation — spot-check pruned
rows against their source turns; the `_OUTCOME_MIN_OBS=4` gate + verified-pin bound the risk).

**Do NOT re-enable an OFF layer** without meeting its criterion or a fresh paired-ablation win.
The default-OFF state is the measured-neutral configuration, not an accident.

**Track B / B3.** Cross-session MEMORY tiers are PROVEN (Track B: 98% recall treatment vs 0%
control). The pure-idle loops (dream/self-play, reflection critique, skills-auto graduation, PRM)
were unadjudicated until B3 — see §4/§6: B3's first live pass (2026-07-07) proved the self-play
loop productive; the deeper "does idle output improve outcomes" question is still open.

---

## 4. WHAT REMAINS TO DO

### 4J. Self-learning stack audit (2026-08-04) — 6 fresh-eye reviews, ~100 findings, CRITICALs FIXED, rest triaged

Six parallel reviewers over the whole self-learning stack: GEPA/optim, reflection+distill,
calibration/confidence, the idle loops, memory learning surfaces, and router/PRM/eval. Every
finding was past a green ~10.7k suite. **The headline is not any single bug — it is that four
separate subsystems the journal recorded as `live` had never run in production.**

**═══ FIXED THIS SESSION ═══**

**THE BIG ONE — a mock-guard that is always False in production.** Six sites used
`type(x).__module__.startswith("ghost_agent")` to mean "not a test double". Prod launches
`python -m src.ghost_agent.main`, so every module is `src.ghost_agent.*` and the guard NEVER
passes; tests run `PYTHONPATH=src` and it ALWAYS passes. Five subsystems were silently inert on
the live agent while their tests stayed green: **failure-cluster distillation** (§3 says `live`),
the **outcome-gated lesson PRUNE**, the **REM project-digest pass**, **file-manifest backfill**,
and the **journal-mined self-play curriculum**. Live proof: `failure_distill_state.json` had never
been created in 16 days, and all 20 records in `selfplay/journal_stash.json` read
`replayed: False`. Fixed via `utils/component_guard.py` (accepts both shapes) + a regression fence
test that fails on any new occurrence.

**Self-play was writing the PRODUCTION calibration corpus and competence prior.** `dream.py`'s
isolated context leaves 29 handles aliased, including `calibration_tracker` and `metacog`, and
`_record_calibration_safe` had no simulation gate even though `is_simulation` is computed 1000
lines earlier. Live: 26 samples in the current epoch carry no `req_id` against 29 self-play runs
in the same window, with the solver's exact tool profile — unfilterable after the fact. The
competence prior (which the confidence composite reads on REAL turns) took hundreds of synthetic
observations a day. Both write paths now carry the same gate.

**Infra failures were charged to the agent as genuine self-play failures.** `docker.execute()`
never raises — it RETURNS `[SANDBOX INFRA ERROR — not your code]`, which became `passed=False`;
and an upstream LLM outage broke the loop on `"CRITICAL:"` with no flag. Either one wrote
`compression_delta=-1.0`, flipped `mastered` True→False, doubled the cooldown, told the
adversarial generator that family is "hard", and could mint a playbook lesson whose only source
material is an outage banner. All durable. Both now route through `validator_infra_crash`.

**Episode outcome labels were an 80-character substring guess.** `"error" not in ai_text[:80]`,
while the verifier verdict and the strike ledger sat in scope unused. Live store: 250/259 (96.5%)
labelled success, **all nine "failures" were false negatives**, and the agent's own total-failure
sentinel was stored as SUCCESS three times. That label feeds the LLM that mints playbook lessons
and gates `search_recoveries`. Now uses the same signals the rest of finalize agrees on.

**Competence observation counts were destroyed on first write.** The legacy mass-derived fallback
was one-shot, so the first `record()` on an old cell wrote `n=1` and threw the history away. Live:
`fs|file_system` reported 889 against 2533 real observations — a number that feeds
`shrink = n/(n+5)` AND is printed into the model's own system prompt. Now `max(counter, mass)`.

**Redaction was corrupting the live corpus.** The `ipv4` rule had none of the structure guards its
`credit_card`/`phone` siblings got: **33 of 41 live `<REDACTED_IP>` hits were not hosts** —
PostgreSQL manual section numbers, a Chrome UA inside stored Python, part of a loopback address.
Fixed with a preceding-context guard; 9/9 cases now correct with real addresses still redacted.

**A quarantined lesson created a permanent dead zone.** Dedup had no quarantine awareness, so a
later CORRECT re-learn matched the quarantined row, returned "reinforced", and was written nowhere
— while `_filter_quarantined` kept the topic out of every prompt, and each bump raised the
quarantined row's utility so `prune_low_utility` could never clear it. Fixed in both dedup
branches; and `_filter_quarantined` now keys on (trigger, solution) rather than trigger alone,
because the first fix exposed a second layer of the same dead zone.

**GEPA (the optimizer that promotes prompts into prod):** `scripts/run_gepa.py` gated against a
STALE baseline and `os.replace`d the live artifact with no backup — a 0.50 candidate beating the
hard-coded 200-char seed would have destroyed the promoted 0.80 `planning.decompose`. Now gates
against the live incumbent, backs it up, and refuses to ship when the private tier is too coarse
to resolve its own `--min-delta`. The generic metric was **recall-only**, making verbosity the
optimum (a token-soup answer scored 0.833 vs 0.333 for a terse correct one, and 0.545 against a
completely unrelated gold) — now token F1, which inverts that ranking. A 2b candidate set could
pass the gate and be **100% inert** because the read-site's aggregate-inflation ceiling was not
checked at the gate. Over-cap candidates were silently graded as the incumbent instead of scored
zero. The activation counter — the ONE instrument built to catch silent inoperativeness —
counted LOADS, not applications, so 6 rejected artifacts read as `applied: 1, fallback: 0`; it now
counts read-site rejections separately. **19% of the GEPA train set was reflection PLANS taught as
gold answers** (505→409 examples after filtering `task_kind`).

**Also fixed:** the post-learning eval suite scored a pure prompt echo 3/5 because `"first step"`
was a discover-keyword and 3 of 5 prompts contain the phrase (measured stub floor 0.600 with zero
agent involvement); the router's heavyweight-tool list named `image_gen`/`vision` against a
registry holding `image_generation`/`vision_analysis`, so the two heaviest turn types never tripped
`hard` (27 mislabelled trajectories) — with a name-drift fence test added; PRM had no finite-weight
check, so a NaN checkpoint loaded silently and returned 0.5 forever (the router already had this
guard and its docstring claimed to mirror PRM); a model-invocable `self_play_loop` could retrain
and overwrite the pinned PRM checkpoint with no consumer gate; the offline eval guard blocked
nothing Ghost actually uses (curl_cffi, subprocess, uvloop all leaked) — widened and its "hard
guarantee" claim corrected; `escalated_overturn` (the §4F FPR watch metric) was never persisted;
and the durable log had **no date**, which is what made an earlier §4F watch window label wrong.

**═══ FOUND, NOT YET FIXED — triaged backlog ═══**

Recorded honestly rather than silently dropped. None is a live-data-loss risk; all are measured.

⚠ **STATUS AS OF 2026-08-04 (later): items 1, 2, 4 and most of 5 are CLOSED, and items 4 and 5
contained FALSE CLAIMS that only a probe caught** — see the round-2 §6 entry. The per-item markers
below are current; read them, not this list's original framing. **Items 3 (router 100% inert) and
the item-4 distill/prune conflict are the live operator decisions.** A defect this round added to
the list: `escalated_overturn` was recorded here as "now persisted" and was in fact persisted into
`VerifyResult.to_dict()`, which has **zero production callers** — fixed via a durable ledger.

1. **verify_bench — CASE POOL ✅ FIXED 2026-08-04; the escalation axis remains open.**
   *Fixed:* the harvest read ONLY the classic prompt, which is the two-stage FALLBACK — 55 of 618
   recorded verify calls, i.e. cases **selected on two-stage failure**. It now inverts the
   ENUMERATE template too (281 records), and the `{{`/`}}` unescape that inversion needs (without
   it, 0 of 580 live records matched while the opening sentence matched perfectly). It now also
   reads the **production verdict** from the same day-file and refuses claims that were REFUTED.
   That join is keyed on the **CLAIM**, not on `request_id` — corrected 2026-08-04 after a review
   showed `request_id` is per-TURN (348 distinct ids over 12,047 records, one `"SYSTEM"` id holding
   9,780). Of the **62** cases the request-level join excluded, **43 were wrong**: a
   `['REFUTED','CONFIRMED']` sequence inside one request is the `_escalate_refute` signature
   (cheap judge false-refutes, main model overturns — all 43 multi-verdict requests are exactly
   this shape), i.e. production did *not* refute those turns and they are the *best* clean cases
   available because they survived two judges. 16 more were dropped on the code auditor's verdict
   about a different claim, and 3 sat in the poisoned `"SYSTEM"` bucket.
   Per-claim, taking the LAST verdict: **106 candidates → 86 kept, 20 dropped**, every kept case
   annotated with its real `prod_verdict` (was: 44 kept, and all 106 annotated `unknown`).
   Known gap, unguarded: a claim REFUTED in its own turn and CONFIRMED later in an *unrelated*
   turn would be admitted as clean — zero instances today.
   Mined cases are now REDACTED before they
   are persisted (day-files are unredacted by design; a case file is durable), which surfaced a
   `credit_card` false positive eating a JSON float — fixed with the same guard the `ipv4` rule got.
   Trials are seeded per **(case, fault)**, so growing the pool no longer rewrites existing trials
   (was 20% of them); only `wrong_topic` can move, and its donor is rank-chosen rather than
   index-chosen — which bounds churn under growth but does not prevent it (for a 21→65 growth the
   incumbent donor is displaced with probability m/(n+m) ≈ 69%; 16 of 21 wrong_topic trials did
   change). The pool hash is what makes that safe, by refusing the comparison. Every report records
   `provenance` (cases sha256, n_cases, duplicate-id count, fault set + hash, judge endpoint/model,
   GHOST_HOME, and the enumerate/adjudicate/**classic** template hashes — the classic one was
   missing and it is the only template the `two_stage_off` arm uses).
   **Result, re-measured 2026-08-04 at the real `--private-pct 30` default: the private tier goes
   4 cases / 30 trials / step 0.0833 → 29 cases / 220 trials / step 0.0093 — finally FINER than
   the 0.02 ship gate.** (An earlier write-up here claimed 7/49/0.0455 → 21/155/0.0132; none of
   those six figures reproduced. The 0.0833 figure is the one that was always right, and it is
   the arithmetic behind the "±0.08 private-gate noise" — meaning the +0.087 ship of 2026-07-30
   was about one trial's worth.) `optimize_verifier.py` now REFUSES to run when the private tier
   cannot resolve its own `--min-delta`, **and loads the mined pool by default** — without it the
   gate ran on the 21 seed cases, resolved to 0.0833, and therefore refused to run at its own
   default flags. `run_gepa.py` has the same guard, now *before* the expensive optimization
   instead of after it.
   The pool lives at `$GHOST_HOME/system/eval/verify_bench_cases_mined.jsonl` (86 cases, out of
   the repo because it derives from live turns); `--no-mined` reproduces the old pool, and
   `scripts/verify_bench.py --refresh-mined` regenerates it — previously **nothing could**, so the
   shipped pool silently kept whatever extraction/redaction bugs were live the day it was minted
   (it was still carrying a `<REDACTED_CC>`-mangled evidence field until 2026-08-04).
   *Escalation axis ✅ FIXED 2026-08-04.* The bench's `HttpChatClient` defined no critic/worker
   route, so `_escalate_refute` returned immediately — the bench scored the RAW cheap judge while
   production scores judge+escalation. **Verified before fixing, two independent ways.** Joining
   recorded verify prompts on the CLAIM across `system/llm_recordings` (07-30..08-04) and reading
   which model served each verdict (worker = Gemma 4 E4B, main = Qwen3.6-35B): **42 of 50 (84%)**
   cheap-judge refutes overturned, per-day 5/8, 17/18, 12/15, 4/5, 4/4. The durable log's
   `GhostAgent` lines over a longer window: **80 overturned / 19 stood = 81%** — the recorded
   figure, reproduced. (⚠ a naive grep of the log gives 89%: OVERTURNED is a WARNING mirrored to
   the `GhostStream` logger while "verdict stands" is INFO, so warnings double-count. Count one
   logger.) An executable probe confirmed the raw client makes exactly ONE call for a REFUTED
   verdict.
   *Fix:* `EscalatingChatClient` mirrors production's topology with no new verifier code —
   `route()` → judge, `chat_completion()` → main (where `force_main=True` lands), truthy
   `worker_clients` so the escalation has a cheap route to escalate FROM, and `route()` degrades
   to `fallback` rather than raising, like `LLMClient.route`. Both legs now carry the verifier's
   OWN bounds (45s route / 90s main); the 90s was arriving via `_bounded_fallback_kwargs` and
   being swallowed. `scripts/verify_bench.py --main-base-url` selects it; verified live
   end-to-end (leg sequence cheap→cheap→main→main on a refuted trial, 24.3s vs 15s).
   *And the report refuses to mix arms:* `score_trials` emits `fpr_raw_judge` XOR `fpr_escalated`
   — **there is no bare `fpr` key any more**, so a stale reader gets a KeyError instead of a wrong
   number; `provenance.escalation` records arm + kill-switch state + cheap route + both
   endpoint/model identities; the raw arm's rendered headline says "NOT a production FPR"; a
   same-endpoint escalation is flagged as an ablation, not production's shape; pre-2026-08-04
   bundles render `arm UNRECORDED` rather than being back-dated to raw. `optimize_verifier.py`
   gains `--escalate {off,gate,all}` (`gate` = production-equivalent SHIP DECISION, training stays
   cheap), refuses `gate`/`all` without `--main-base-url`, and writes `gate_arm`/`train_arm`/
   `gate_judge`/`gate_main` into every promoted artifact. Tests:
   `tests/test_verify_bench_escalation_arm.py` (the arm label is fenced against OBSERVED call
   sequences through the real `Verifier`, so a change to the escalation predicate fails a test
   instead of mislabelling a report — both mutants caught) + `tests/test_optimize_verifier_arm.py`.
   *Docstring claims corrected while in there (all re-measured):* `verify_bench.py` pointed
   `--base-url` at `127.0.0.1:8080` and documented `nova:8081` — **nothing listens on either**;
   the live judge is `100.83.184.117:8088` (Nova/E4B) and the main model `127.0.0.1:8088`
   (Qwen 35B). `balanced_score` claimed a "~5:1" refute-heavy trial mix — the T0 bundle it refers
   to is **3.41:1** (75/22), and today's pool is 3.06:1 public / 3.07:1 private. It claimed "~18
   escalation overturns/day" — measured **~8/day** (42 over 5 days); 18 was the peak day quoted
   as an average. The cap-guard comment claimed "100% of its refutes overturned by the 35B" —
   that was **one day at n=4**; corpus-wide it is 84%. (Confirmed still true: 21 seed cases, and
   the 4 cases / 30 trials / 0.0833 → 29 / 220 / 0.0093 private-tier figures reproduce exactly.)
   Evidence size is no longer a gap (mined median 749, max 3998 chars, against a seed median of
   198 and a production median of 3291).
   *Both directions ✅ 2026-08-04 (same session, after item 2 landed).* The CONFIRM escalation
   re-opened the identical gap in the other direction: it fires only on `high_stakes=`, which
   `run_trials` never passed, so it was structurally dead in the bench even with the two-leg
   client installed. Bench cases now carry a **tri-state** `high_stakes` (`None` = derive, an
   explicit bool pins — `bool(obj.get(...))` would have collapsed absent into False and frozen
   the direction dark), derived by running production's OWN `looks_like_tool_error` over each
   SEGMENT of the packed evidence digest. Segmenting is load-bearing: that sniffer scans only the
   first 120 chars for its text markers, so a blob check sees just the first tool's head —
   measured on the 86-case mined pool, **segmented 14 (16.3%) vs blob-only 10 (11.6%)**. Honest
   bound: the digest is budget-truncated, so this is a LOWER bound on production's flag.
   Resolution is per trial AFTER the fault, which is what makes `silent_failure` (a tool error
   under an unchanged success claim — exactly `_escalate_confirm`'s population) reach the
   direction: 0 of 21 seed cases are naturally high-stakes, but 70 of 107 `silent_failure` trials
   are — not 107, because one of the three injected bodies is `(empty output)`, which production
   does not class as an error either.
   *Arm labelling now has FOUR values*, because the directions are independent and a run with one
   live is production-equivalent in neither: `raw_judge`, `judge+escalation(refute)`,
   `judge+escalation(confirm)`, `judge+escalation`. A stale reader comparing
   `arm == "judge+escalation"` therefore FAILS on a half-escalated run instead of accepting it.
   New metric `false_confirm_actionable_{raw,escalated}` (corrupted trials CONFIRMED, plus the
   ≥0.7 actionable rate — item 2's headline quantity, which the bench reported NOWHERE before).
   **Each metric is keyed on the direction that can actually move it**, not on the whole arm:
   `_escalate_confirm` provably never emits a REFUTED (it returns the main model's CONFIRMED or
   caps the cheap one to 0.6), so it cannot move FPR/TPR — keying FPR on the full arm would be
   false precision blocking a legitimate comparison. `metrics["escalation_events"]` counts
   overturns/withholds READ OFF the verdicts, and a live confirm direction with zero high-stakes
   trials is called out as measuring nothing. `GHOST_VERIFY_ESCALATE_CONFIRM=0` is verified by
   A/B (same verdict, same confidence, same call sequence), not asserted. Three mutants caught
   (dropped `high_stakes` kwarg, confirm-arm mirroring refute, blob-only derivation).
   ⚠ **Known and deliberate:** the confirm direction CANNOT move `optimize_verifier`'s gate
   metric — `_trial_score` is verdict-only and a withheld confirm changes only confidence — so
   under `--escalate gate` it costs one main-model call per high-stakes CONFIRMED private trial
   for zero signal. Kept anyway (a gate measuring a different pipeline from production is the
   defect being closed); making the score actionable-confidence aware is a real design decision,
   flagged rather than slipped in.
2. **The cheap judge's CONFIRMED direction has no escalation** — ✅ **FIXED 2026-08-04 (later)**,
   and the live measurement is sharper than this entry was: **0 of 130** cheap verdicts sit below
   the 0.7 gate (0.9×51, 1.0×79), so the gate filters NOTHING and every cheap CONFIRMED was
   consumed unconditionally while every REFUTED got a strong-model check. `_escalate_confirm` now
   mirrors `_escalate_refute` on HIGH-STAKES turns only (a tool failed this turn — exactly when the
   verdict is what keeps the turn out of a FAILED label). **Deliberately not symmetric:** a withheld
   confirmation does not flip to REFUTED (that path is punitive — auditor note, lesson retraction,
   FAILED label, auto-repair); it keeps CONFIRMED and caps confidence at 0.6, below every
   consumption gate, so the turn is recorded UNVERIFIED rather than either fabricated-passed or
   manufactured-failed. `GHOST_VERIFY_ESCALATE_CONFIRM=0`, default ON. Traced consequence: on the
   narrow intersection (high-stakes ∧ withheld ∧ last action an untested write) the auto-repair
   predicate already treats `CONFIRMED && conf < 0.7` as "unverified mutation", so one bounded
   repair round now fires — coherent, but a behaviour change beyond the label. Residual risk: an
   honestly-reported failure the 35B declines to confirm lands structural-FAILED, narrowing the
   2026-07-31 operator rule to "honest AND strong-judge-confirmed"; live incidence of that exact
   upgrade is 2 turns in 28 days and the replay flipped 0 honest-report cases.
3. **The router is measurably good (CV lift +0.138 over majority, 837 live decisions) and 100%
   inert** — both consumers are dark (`_MCTS_TURNSTART_ENABLED = False`; `use_planning` is not an
   argparse argument at all). §4I's premise that it "gates MCTS + the planner" is false.
4. ~~**`failure_distill`'s gate is structurally unreachable**~~ ❌ **THIS CLAIM WAS FALSE —
   disproven by probe 2026-08-04 (later).** The live `failure_distill_state.json` shows **three
   clusters fired on 2026-08-04** (`model/python_general` 4 cases, `output_processing` 3,
   `orchestration` 3) producing 2 `source=distilled` playbook rows. Real arithmetic over the
   production `gather_failure_corpus`: 42 playbook rows → 27 dropped (11 mistake-less, 14 outside
   14 d, 2 already distilled) → 15 + 4 work_log = **19 corpus records**, **37% unattributed (not
   69%)** *before* the LLM adjudication that runs first in production, **6 groups**, **2 at/over
   `_MIN_CLUSTER`=3**. The gate is reachable and has fired. ✅ The REAL defect, now fixed: a pass
   producing nothing was indistinguishable from a pass that never ran — every cycle stamps a
   reserved `_last_run` key and logs when barren for a STRUCTURAL reason (`empty_corpus` /
   `no_cluster_reached_threshold`), with "all clusters unchanged" left at debug as the healthy
   steady state. Trade-off documented: the file's existence is no longer proof a lesson was written
   — the per-cluster keys are.
   ⚠ **The distill/prune conflict is real and now QUANTIFIED — operator decision, not a code
   cleanup.** Live bottom-quartile cutoff **1.0716**; the two distilled rows score **0.3633**
   (7 retrievals — already prune-eligible one day after minting) and **0.7943**. Cause: broad
   pattern lessons hydrate often, are rarely marked helpful, so the stale penalty halves them.
   `GHOST_SKILL_PRUNE` stays OFF, so nothing is being destroyed today.
5. Frontier compression signal is arithmetically dead at the length floor (all 8 live clusters at
   `best_length=4`, **1** positive delta in 200 runs — this entry said 2; census 83 zeros, 116
   negatives). CONFIRMED dead, deliberately NOT changed: making it informative means changing WHAT
   is measured (solution tokens / AST size / the existing novelty score), an operator decision.
   ✅ **Mastery-by-duplicate FIXED 2026-08-04 (later)** — the dedup branch protected
   `runs`/`total_first_try_wins`/`best_length` but mastery is decided from `recent_outcomes`, which
   duplicates joined; live, `regex_parse` had **10 of 10** recent outcomes flagged duplicate, one
   real run from mastering on pure re-rolls. The streak now filters `duplicate=True`, and this
   cannot recreate the "pinned unmastered, re-picked forever" pathology because duplicates carry
   `passed=True, delta=0.0`, keeping `_cluster_is_saturated` true (test pins it).
   ✅ `Trajectory.cluster` **confirmed None on 1488/1488**; its only consumer (`--frontier-selfplay`)
   is off in the live config and was flipped off on an ablation verdict, so no producer was added —
   instead `count_trajectories_by_cluster` now WARNS on N trajectories with zero clusters, making
   the constant visible the moment anyone re-enables the flag. The reflection branch bypassing
   mastery/saturation filters is still open.
6. skills_auto injects its "PROVEN APPROACHES" block on **83.6%** of turns (overlap > 0 on
   3-char tokens); its cooldown is in-memory so restarts bypass it; re-graduation is reported as
   new learning (251 digest lines for 5 skills, none new since 07-31).
7. 94% of live lessons carry no `source_trajectory_id`, so `retract_lessons_from_trajectory`
   cannot reach them while printing a "scrubbing" banner; dedup-reinforce never restamps
   provenance.
8. λ = 0.4 is live and ungated by the separation test (bought +6.9e-05 Brier on 7 rows; any single
   row dropped flips it back to 0); the "unverified mutation" backfill mints the hard 0.0
   "checked and WRONG" label for turns nothing checked; Tier-2 negatives drop `req_id`+`domain`,
   so the §4H triage cannot see the strongest negatives and Tier-2/Tier-3 can double-count one turn.
9. All four calibration diagnostics (`brier_score`, `reliability_table`, `ece`, `stats`) have ZERO
   callers, never apply the Platt map, and measure a column written under ≥6 different formulas —
   live `ece` reads 0.0459 against a true 0.0023.
10. Graph edge `weight` is a re-assertion counter, not evidence: 183 of 823 live edges are already
    permanently immortal, and re-assertion resurrects soft-expired edges.
11. Self-play's tool containment is a 12-name denylist against a 40-tool registry — 28 survive,
    including real Tor egress and real-state writers (`notify_operator`, `manage_projects`).
    Should be an allowlist.
12. ~30 further MEDIUM/LOW items and **~45 docstring claims measured false** across the six
    reports, including several numeric claims that have simply drifted.

**Method note worth keeping:** every one of these six reviews was told to write executable probes
and spot-check numeric docstring claims rather than trust them. That instruction is what produced
the module-guard finding, the 96.5% episode mislabel, and the 33/41 redaction corruption — none of
which is visible by reading the code as written.

### 4I. Router confidence — the last unused prospective signal (2026-08-05) ⏳ PHASES 1-3 ACTIVE, PHASE 4 GATED

**What it is.** `complexity router: hard (confidence 0.10) ESCALATED` is computed on EVERY request
at turn 0 (`core/agent.py` ~12460, `router/dispatch.py`), gates MCTS + the strategic planner, is
written to the durable log — and is then discarded. It is the only **PROSPECTIVE** difficulty
signal in the stack: available BEFORE the turn runs, where the depth curve (§4H) is retrospective
and cannot say anything before step 6.

**Audit that found it (2026-08-05).** Checked which recorded-but-unconsumed signals exist:
| signal | state | note |
|---|---|---|
| router confidence/label | computed every request, **never stored** | this section |
| `Trajectory.tokens_in/out` | **0 of 183** recent trajectories populated | dead fields; upstream carries usage |
| `Trajectory.validator_signal` | 0 of 183 populated | dead on chat turns |
| `repair_round` | drives the repair loop, never a feature | CONFIRMED trouble (verifier refuted), unlike depth's prior |
| UNKNOWN rate | now monitored as an attrition confound | latent LABEL TIER: "stopped and reported honestly" ≠ "silently failed" |

**Phase 1 — make it durable (no behaviour change). ✅ SHIPPED 2026-08-05.** `router_label`,
`router_confidence` and `router_escalated` are stamped into `Trajectory.extra` at the decision
site, via a new `core/turn_facts.py` — a bounded req_id-keyed ring, because the streamed path
writes its trajectory AFTER the semaphore is released and a plain context attribute belongs to
whichever request is running then (the exact defect found live on the experiment-arm stamp; the
module is that lesson extracted so the next mid-turn fact does not rediscover it). Deliberately NOT
folded into `core.experiments`' ring: that one holds only ENROLLED requests, so facts would vanish
whenever the framework was killed.

**`tokens_in`/`tokens_out` NOT done — deliberately, with a reason.** They read 0 of 183 on recent
trajectories, but nothing in the stack captures upstream `usage` at all (`grep '"usage"'` over
`core/llm.py` + `core/agent.py`: no hits), so filling them means touching the LLM response hot
path. That is a bigger change than a recording-only phase should carry. Left as its own item.

**Phase 2 — VALIDATE BEFORE TRUSTING. ✅ INSTRUMENT BUILT 2026-08-05
(`scripts/router_confidence_backtest.py`), verdict awaits supply.** Buckets stamped turns by
turn-0 confidence (keeping the router's own 0.30 escalation threshold as a bucket edge — if the
signal is real the split should be visible right there), reports each bucket's actual failure rate
with an anytime-valid interval, and exits 1 unless the spread across usable buckets clears 0.10.
Run against the live corpus today it correctly reports `0 carry a router stamp` — Phase 1 has only
just deployed, so there is nothing to measure yet. Re-run after ~a week of traffic. Full design: The same backtest the
depth curve had to pass: bucket historical turns by router confidence, measure the ACTUAL failure
rate per bucket. Two outcomes, different destinations — if it discriminates (e.g. 15% failure at
high confidence vs 45% at low) it is a real prior; if it is flat, it is an escalation heuristic
that happens to be logged, **and the plan stops at Phase 2**. Do not skip this: the depth curve
earned its place because 342 trajectories said so, and this project's history is full of signals
that looked usable and measured dead (`uncertainty_pressure`: 2 distinct values across a whole
epoch; competence: separation 0.0023).

**Phase 3 — variance reduction. ✅ BUILT 2026-08-05, and the expected size was OVERSTATED.**
Uses the stamp as a PRE-TREATMENT covariate (CUPED) in `compare_arms`: pooled
`theta = Cov(Y,X)/Var(X)`.

⚠ **This paragraph was WRONG in three places and is corrected here** (caught by a later review
reading the code against the journal). What ships:
- the adjustment moves the **reported means as well as the interval** — the re-centring IS the
  variance reduction, and keeping the raw mean while adopting the shrunken width produced 12/300
  false verdicts at zero true effect (0/300 after the fix);
- a widening covariate is **adopted, not discarded** — `min(raw, adjusted)` is the
  better-of-two-draws effect, measured consistently anti-conservative for no benefit; only the
  REPORTED `variance_reduction` is floored at 0;
- "safe by construction" is too strong: `theta ~ 0` does make a useless covariate a no-op, but the
  covariate is pre-treatment WITHIN a request only — the router sees the previous (possibly
  treated) assistant message, so there is second-order leakage across a conversation. The code
  says so; this section did not.

⚠ **MEASURED BENEFIT, correcting my own earlier claim.** I wrote "typically cuts variance 20-50%".
That figure comes from CUPED's usual setting — a pre-period version of the SAME metric, correlation
0.6-0.9. A turn-0 difficulty score against a BINARY outcome cannot reach that. Measured on
synthetic data at n=200/arm:

| covariate strength | CS half-width reduction |
|---|---|
| none | 0.3% |
| weak (0.2) | 0.0% |
| moderate (0.4) | 1.3% |
| strong (0.6) | 4.1% |
| very strong (0.8) | **8.6%** |

So: a real but SMALL win, it costs nothing, and `MetricComparison.variance_reduction` reports what
it actually bought on every line rather than asserting a benefit. The 20-50% figure would need a
covariate of a different KIND — a per-session prior failure rate, or a pre-period metric — which is
worth looking for if time-to-verdict ever becomes the binding constraint it looks like today.

**Phase 4 — an early gate. GATED ON PHASE 2 SAYING IT DISCRIMINATES. Do NOT start otherwise.**

*The gap it fills.* The risk governor cannot fire before step 6 (`_MIN_STEER_STEP`), and by
construction it cannot: its two moving terms are turn shape and the strike ledger, both of which
need a turn to have already happened. A request that is hard **from the first token** therefore
gets no adaptation at all until it is already six steps deep and ~42% likely to fail. Router
confidence is the only signal that could act at step 0.

*What it would do.* One of two treatments, chosen by what Phase 2 measures — NOT both:
- **budget** — a router-hard request gets a larger turn budget and/or a raised
  `_MIN_STEER_STEP`, on the theory that hard tasks legitimately run long and the current gate
  punishes them for it (this is the "mechanism, not outcome" worry in reverse: if the steer's
  effect is mostly "ends turns earlier", a hard request is exactly where that is harmful);
- **early frame** — a gentler turn-0 nudge than the step-6 steer: state the plan, name the
  single riskiest assumption, and say what would falsify it. Deliberately NOT the "STOP and
  report" text, which is wrong at step 0.

*Ship rules, non-negotiable:*
1. It ships as its OWN experiment arm (`router_budget` or `router_frame`) in
   `core/experiments.py` — never as a default. "Obviously better" is the claim that keeps being
   wrong here (two GEPA verifier rounds rejected by the private gate; the earn-keep harness
   pruning zero subsystems).
2. It must register a compliance key in `TRIGGER_KEYS` so the powered (triggered-only) comparison
   exists, and — if it mutates the prompt — in `CONTEXT_MUTATING_KEYS`, or it silently
   contaminates the §4F Phase 2b fixture corpus.
3. Turn-0 assignment means the treatment applies to the WHOLE turn, so `n_steps`/`duration_s`
   are mechanism metrics for it too. Read `failure_rate` on the triggered block, and read the
   `unknown_rate` line above it first.
4. **Two live steers must not stack.** If `router_frame` fires at step 0 and the risk governor
   fires at step 6, that is two SYSTEM ALERTs in one request — the exact thing the governor
   already yields to the futility breaker to avoid. Decide the precedence BEFORE building, and
   make the deference explicit in code rather than in a comment (a one-way claim in a comment
   was already found false once, §6 2026-08-05).

*Kill criterion.* If the triggered comparison shows no effect after the arm reaches n>=30/arm on
both sides, retire it rather than leaving it default-off in the tree — the "built but unwired"
inventory is already long enough.

### 4H. Confidence-score follow-ups (2026-08-02) — 2 items, both OPEN, neither blocking

Left over from the calibration epoch/objective fix (§6 2026-08-02, DEPLOYED + live-verified). Both are
deliberate non-actions with a stated trigger, not oversights — **read the §6 entry before starting
either**, especially the measured verdict that the score has no behavioural consumer today
(`_METACOG_ARBITER_ENABLED = False`) and that a mid-turn confidence gate is NOT worth building.

1. **`λ` is not gated by the separation test.** `_MIN_SEPARATION_SIGMAS = 2.5` gates `w_entropy` and
   `w_effort` — a weight may only leave zero once its feature separates the outcome classes by 2.5
   standard errors. The `λ` (verbalised-uncertainty) grid is still searched unconditionally, so it is
   the one free parameter the delivered-Brier objective can buy without evidence.
   **Why it is safe to leave:** `uncertainty_pressure` is dead — 2 distinct values across the whole
   live epoch, mean 0.0003, max 0.067 — so `(1 − λ·pressure)` can move a composite by at most ~3%,
   and only on the handful of rows carrying any pressure. The 2026-08-02 refit fitted `lam=0.00`
   anyway. Gating an inert term now is churn.
   **Trigger to act:** the moment `UncertaintyTracker` starts producing real signal (watch
   `uncertainty_pressure` distinct-values / separation in `introspect action='learning'`). At that
   point λ would be fitting a LIVE feature that never had to earn its place. Fix = same one-line
   `and _separation_sigmas(...) >= _MIN_SEPARATION_SIGMAS` conjunct the other two weights use.
   Documented as "Known asymmetry" in `docs/core/calibration.html`.

2. **Offline triage — ✅ BUILT 2026-08-05** (§6 that date). Reflection now ranks a bounded pool
   worst-first instead of taking the OLDEST failures. ⚠ NOT as designed here: ranking by the
   calibrated score *where it exists* and the shape proxy elsewhere mixes two incommensurable
   scales and measured WORSE than either alone (Kendall τ 0.605 vs 0.766 shape-only / 0.763
   calibrated-only); on the live corpus it DEMOTED 7 of the 8 joinable failures. Shipped as ONE
   scale always — calibrated only at full pool coverage (~5% join rate today, so shape in
   practice). The design intent below is kept as the record of what was asked for.
   The original plan: rank trajectories
   for reflection / postmortem / self-play attention by the calibrated post-hoc score (**AUC 0.727**
   on 342 labelled trajectories) instead of the binary "failed" flag those selectors use today. A
   graded calibrated probability is strictly more informative than a boolean, it is exactly where the
   signal is strong (process health), and it touches no live path — so it needs no §3-style
   measured-win gate before shipping.
   **Design call, not a defect** — hence parked rather than done. Two caveats to carry in: the label
   is a PROXY (`grade_turn_outcome`), so this ranks *process health*, not correctness; and the
   ground-truth tiers are thin (5 of 546 samples are `failure_report`/`task_reopened`), so keep them
   flowing or the ranking drifts toward rewarding "did not visibly break".
   **Free adjunct needing no model at all:** turn DEPTH predicts failure on its own — 17.8% at step 1,
   35.6% at 4, 42.3% at 6, 52.0% at 8, 60.6% at 12. Usable for budget/escalation policy directly.
   ✅ CONSUMED 2026-08-05 by `core/risk.py` (the live steer quotes this curve; the blend around it
   is explicitly a heuristic ordering, not a probability). Backtest: the gate fires on 7.6% of user
   turns and those turns fail at 0.543 vs a 0.277 baseline.

### 4G. Project-aware services + port leases — 3-phase plan (2026-07-30) ✅ IMPLEMENTED same day

> **Status 2026-07-30 (evening): all three phases SHIPPED** — see the §6 session entry for the
> as-built record (allocator grants a lease on EVERY start unless `port=0`; adopt action;
> scoped keys `<project>:<name>`; boot reconcile line; SERVICES briefing block; execute
> daemon guard). NOT yet deployed to the live agent at write time.

**Problem (operator, after the solar-sim guard incident):** many projects reach for the same port;
the agent has no record of which project uses what; unregistered daemons (execute `… &` leaks) hold
ports invisibly; a start on an occupied port thrashes instead of moving. Requirements: NO service
auto-restart on agent restart (current behavior — keep); occupied preferred port → auto-assign a
different one; multiple projects+services concurrently; agent aware of all of it. Services expose a
project's output on demand and are optional per project.

**Design: services are project-owned resources; ports are LEASES granted by the supervisor, never
numbers the model picks.** The PORT/HOST env-export contract already exists — only the allocator is
missing. Registry (`/workspace/.services/registry.json`) becomes the single source of truth for
port ownership; briefings surface it; reconciliation keeps it honest against `ss -ltnp` reality.

- **Phase 1 — ownership + leases:** `project_id` stamped on registry entries from the bound project
  (string-heuristic `_project_service_entries` becomes legacy fallback); names project-scoped
  (`<project_id>:<name>` key); port allocator — omitted port → first free published port (no live
  lease + in-container bind probe); given port = PREFERENCE → if leased/held, auto-fallback to next
  free port and SAY SO (never kill the holder); literal-port substitution in commands when the
  requested port moves; lease release on stop/stop-all/archive/delete.
- **Phase 2 — reconcile + awareness:** read-only `reconcile()` (registry ⋈ actual listeners) at
  boot (one pretty_log line, NO restarts), before allocation, and on status — surfaces
  listening-but-unregistered orphans (pid, cmdline, project inferred from /proc/<pid>/cwd) with
  adopt-or-kill; ACTIVE-project briefings get a SERVICES line (RELEASED already has one); project
  list shows services+ports; global status gains a port map; tool doc: "omit port to auto-assign".
- **Phase 3 — close the orphan source + capacity:** execute-tool steer for backgrounded listeners
  (`… &`/nohup/setsid + server patterns → use manage_services; the solar-sim leak was autoadvance
  running `python3 server.py &`); autoadvance routes server-start steps through the supervisor;
  derive MAX_SERVICES from the published-range size (both are 5 today, silently coupled); document
  that widening GHOST_SANDBOX_SERVICE_PORTS needs a container recreate (which kills services).

**Not changing:** no auto-restart at boot (reconcile is read-only; restart stays one explicit
action using stored command/port/workdir); services stay optional (`project_id` nullable);
container-recreate deaths stay honestly reported (generation stamp, existing).

> **╔═ 2026-07 BUG-HUNT CYCLE CLOSED (2026-07-22) ═╗**
> The dedicated adversarial-hunt cycle is complete. Cohorts swept, fixed, tested, and DEPLOYED this
> cycle: **project-autonomy + turn-loop + code-correction** (07-20 three-stack), **metacognitive stack**
> (07-20), **LLM / routing / delegation / recording / grammar / consumers** (07-22), **memory
> substrate** — vector/graph/episodes/bus/journal (07-22 "later 3"), and **sandbox/execution + infra**
> (07-22 "later 4"). Every finding from those reviews is either FIXED+deployed or explicitly parked
> below. The detailed §4A–§4D catalogues below are kept as the historical record; their per-item
> "FIXED"/"RESOLVED" markers are current.
>
> **What genuinely remains is NOT pending hunt-work — it is three buckets:**
>
> **(0) ✅ negative-label supply COMPLETE through Tier 3 (§4E).** Tiers 1-2 (graded labels +
> provenance; failure-report detection) DONE 2026-07-27; Tier 3 (reopened-work retro-negatives,
> `task_reopened`) DONE 2026-08-01 — marginal supply by design, cost one column + one hook. Tier 4
> stays on HOLD pending a different instrument. Measure a signal's SUPPLY before designing its
> detector: doing so already killed two of my own designs (see §4E).
>
> **(1) Blocked on operator action** (cannot be done headless):
> - **Earn-your-keep / synthetic-ablation route — CLOSED as INCONCLUSIVE-for-this-model (operator decision
>   2026-07-23; §6 "later 3").** DO NOT resurface as pending work. The self-measuring→self-pruning premise
>   assumed auto-graded deterministic tasks could discriminate whether a subsystem helps. They can't on this
>   uncontended 35B: Track A puzzles ceilinged and Track B4 grounded tasks ceilinged (32/35 at 3/3; the 3
>   "survivors" were pure timeout flakes) — the statistical auto-prune rule will never fire on this instrument.
>   **Verdict: ZERO subsystems pruned; prod config unchanged (everything stays on).** What we DID establish:
>   the per-turn stack in aggregate ≈ stripped on all measurable tasks (neutral, not harmful), and the verifier
>   is exonerated (confirms 100%, never false-refutes). The harness code (`scripts/earn_keep.py`, the LOO
>   matrix, `prune_overrides`, the `--no-dream`/`--no-self-play` gates) stays in the tree as dormant infra —
>   not deleted, revivable only if the *instrument* changes (observational mediation on live trajectories, or
>   a deliberately degraded model). Not a headless-runnable item any more.
> - **GAIA full run** — harness done, pilot 8/8; blocked on `huggingface-cli login` for the gated set.
>
> **(2) Parked by decision** (reviewed, deliberately not fixed):
> - **Node-deployed items — SKIPPED per operator 2026-07-22 ("not in use"):** voice-server no-auth CRIT
>   (Orin), uConsole client (SSE token-loss / camera leak), Slack thread-boundary. Reviewed with fix
>   recipes in §4B; per-device deploys.
> - **Linux-only** (prod is macOS/bridge, no prod impact): sandbox `HOST=0.0.0.0` host-mode exposure
>   (accessor in place), exec-user provisioning, egress-guard loopback bypass.
> - **Low-value improvements + one deferred refactor:** llm.py node-payload serializer unify, verifier
>   stage shared-prefix cache, fallback-hint merge, streaming recording hook (dev feature), and the
>   agent.py step-4 streamer-closure refactor (§4A #5 — the seam is established; cosmetic, high-effort).
> - **Latent §4B/§4C residuals** — no prod caller / multi-process-only / model-behavior edges, kept for
>   reference (e.g. remove-while-exec self-heal, Yandex-over-Tor, huge-reasoning no-file-spec).
>
> Next cycle, if reopened, the stalest un-hunted surfaces are the web-facing `interface/server.py` + CLI
> and the mid-July tool cohorts. **╚════════════════════════════════════════════╝**

The detailed catalogue follows. **§4F and §4E are the ACTIVE items (§4F added 2026-07-29); A-D are
historical record whose per-item FIXED/RESOLVED markers are current.** Grouped: (F) the 4-phase
agentic-methodology upgrade plan (ACTIVE), (E) negative-label-supply tiers 2-4 (PENDING), (A)
improvement-review partials/blocked, (B) static-hunt deferred findings, (C) functional-hunt deferred
findings, (D) the B4 outcome-battery design.

### 4F. Agentic-methodology upgrade — 4-phase plan (2026-07-29) ⏳ ACTIVE
### ▶ START HERE: the "NEXT STEPS — §4F, 2026-08-04" block below is the current work list.
### Phase 0 DONE · Phase 1 done-but-de-recorded · Phase 2a shipped · Phase 2b supply-blocked ~08-17 · Phase 3 built/default-OFF · Phase 4 parked

**Origin.** Four-agent deep web survey of mid-2026 agentic-engineering SOTA vs this stack (memory:
`agentic-methodology-survey-2026-07`). Three cross-validated heavy-impact gaps: (1) text-space
prompt/skill optimization (GEPA-class — ICLR 2026 oral, beats weight-level RL at 35× fewer rollouts;
SkillOpt measured **+9.1 on Qwen3.6-35B-A3B**, our exact model), (2) trajectory-level verifier-guided
test-time scaling (+12.1 avg on an 8B — the largest fixed-model deltas of the period), (3) long-horizon
context discipline (proactive timed injection +8.3 TB2; structured compaction +46% for small LMs).
**Discovery that reordered the plan: the GEPA loop is FULLY BUILT and read-wired but has NEVER RUN** —
`optim/` (signatures/trainset/run_gepa/ab_eval/loader), read-site live at the planner
(`core/agent.py` ~12313), but no `$GHOST_HOME/system/optim/` exists and 24 days of trajectories sit
unused. Phases 0-2 close that loop; Phase 3 is the only net-new machinery; Phase 4 is deliberate last
(turn-loop surgery).

**Phase 0 — eval hygiene (DONE 2026-07-29, this session).** Prereq for ANY optimizer: reward hacking
hits 46-74% of self-optimization runs and RISES 26%→58% from 10→100 steps; self-critique doesn't fix
it; a hidden holdout does (63%→34%, AIDE²). Shipped: (a) per-item deterministic PUBLIC/PRIVATE split
(sha256 of stable example identity, default 30% private) — membership can never migrate as the corpus
grows, unlike the old seeded-positional `split_train_eval` which re-deals membership every run (slow
leakage); optimizer + GEPA-internal val see ONLY public, the A/B ship-gate judges ONLY private;
(b) `MAX_OPT_ITERATIONS` hard cap clamped at the `run_gepa()` chokepoint; (c) activation telemetry —
loader counts tuned-vs-baseline applications per signature, surfaced in learning-health (the 2026
"harness updating ≠ harness benefit" result: mid-tier models gain most but the dominant failure is
the component never firing; our own built-but-unwired history says the same).

**Phase 1 — ignite the existing GEPA loop. ✅ DONE 2026-07-29 (§6 "later 7") — CANDIDATE PROMOTED.**
Ran on `planning.decompose`, NOT tool_selection.pick as planned: the generic trainset
(user_request→final_response/plan) is field-coherent only with the planning signature —
tool_selection.pick needs a Phase-2 extractor for step_description/tool_catalog examples. Result:
GEPA public valset 0.401 → 0.589; **A/B on the 20-example PRIVATE tier: baseline 0.45 → candidate
0.80 (delta +0.35) → promoted** to `$GHOST_HOME/system/optim/planning.decompose.json`. The 35B
self-reflector did NOT plateau — it discovered strict output structure + explicit constraint-encoding
("[Action] while strictly adhering to [constraint]") from Ghost's own traces. Ignition flushed 6
written-but-never-run defects — see §6.
**Post-restart activation check (2026-07-29 20:38): counter correctly reads 0 — the planner
read-site is DARK in prod.** Live exec line (ps eww) has NO `--use-planning`; the strategic-planner
path (agent.py `use_plan` gate) is the ONLY consumer of planning.decompose/tool_selection.pick tuned
prompts. The telemetry caught exactly the defect class it was built for, on day one. Disposition:
artifact stays (loads the moment planning is enabled); do NOT flip `--use-planning` without a paired
ablation per §3 doctrine (a better prompt for a subsystem is not evidence the subsystem earns its
latency); Phase 2 therefore targets read-sites that are LIVE in prod — verifier prompts (verifier
ENABLED since 2026-07-05) and tool descriptions (always live).

**Phase 2 — extend to the two highest-leverage surfaces. (a) verifier prompts ✅ SHIPPED+LIVE
2026-07-30 (§6):** optimized against verify_bench via a custom gepa adapter over the REAL two-stage
pipeline; private gate +0.087 (0.796→0.883, n=23 never-seen trials); deployed via restart; both
templates confirmed loading on live turns. (b) tool descriptions — ⏳ COLLECTING FIXTURES since
2026-07-30 ~16:06: the §4E supply-first check measured ZERO usable fixtures (recordings off since
07-17; the one day-file is a stub) — so `GHOST_LLM_RECORD=1` is now exported in the launcher
(dated comment + removal criterion; ⚠ recordings are UNREDACTED, local-disk only) and verified
capturing. **SUPPLY AUDIT 2026-08-01 over the first full day (07-31, 3015 records / 68 MB): all
required data IS being captured — miner contract CORRECTED.** The main agent loop STREAMS: tool
choices ride `kind=chat_completion_stream` records whose reassembled responses carry STRUCTURED
`tool_calls` (327/399; the 07-30 "content-embedded" note was the non-stream path only — those 17
records have empty content and are ignorable). Distribution: file_system 134, browser 114,
vision 31, execute 25, manage_projects 16, manage_services 13, web_search 12 (+2 minor) over 66
requests ≈ **~330 fixtures/day → ~2k by T+7d**. Choice→result pairing WORKS: 319/327 choices link
to their outcome via ordinal-consecutive records in the same request_id (tool results ride
USER-role messages in this dialect). Miner gotchas pinned: outcome classifier must treat
`EXIT CODE: 0` as success (bare "EXIT CODE" match overcounts errors 136 vs true subset) and use
the failure-report conventions; split by request_id hash (session_id also present).
**CONTRACT AMENDMENTS from the operator's 2026-07-31 fix cluster (audited 2026-08-01):**
(1) **ERA FILTER — use only records with ts ≥ 2026-07-31T19:15 local**: the
QWEN_TOOL_PROMPT_NATIVE split (§6 07-31 "later 2", deployed ~18:54) changed the system prompt in
every native-path context, and the honest-failure rule landed ~19:14 — earlier records embed the
old dual-dialect prompt and old outcome semantics (current-era supply: 87 on 07-31 evening + ~330
per full day thereafter). (2) **GROUND TRUTH = trajectory outcome labels via
TrajectoryCollector.iter_trajectories() (corrections-sidecar overlay applied), joined on
request_id — NOT raw exit-code heuristics**: the honest-failure rule (§6 07-31 "later 3") means a
failed tool + honest report = PASSED; for tool-CHOICE polarity, honest-failure turns are
EXCLUDED from the fixture set (choice signal ambiguous — never labeled bad per the rule, not
claimed good either), clean PASSED = positive, REFUTED/shape-FAILED = negative. RUN PLAN
otherwise unchanged: mine → registry read-site via optim loader → gepa → private gate → restart
deploy → FLIP GHOST_LLM_RECORD BACK OFF + archive/delete day-files per operator preference. The `optim/__init__` exclusions (dream/watchdog/safety prompts) STAND.

**Phase 3 — trajectory-level test-time scaling. ✅ BUILT 2026-07-30 (§6), default-OFF pending
measured wins.** (a) Logit-expectation probe: BUILT as a score-token probe (digit-scale + top-
logprobs expectation — the verdict path has no logprob access; the worker route() returns content
only), `GHOST_VERIFY_LOGIT_EXPECT`, benched via the A/B in §6. (b) Adaptive comparative best-of-N:
BUILT (`core/tts.py` + loop-exit hook), trigger = verifier WOBBLE BAND (UNCERTAIN or sub-0.7
REFUTED — not the confidence composite; hard REFUTED keeps auto-repair), `GHOST_TTS_ADAPTIVE_BON`.
(c) Verified-restart: RESOLVED AS PRE-EXISTING — the auto-repair loop IS critique-conditioned
restart with narration discard + round caps; summary-conditioning delta deferred as marginal.
STILL TODO before any live enablement: ~~the stable-prefix-hash regression test with Phase-3
features active~~ (✅ DONE 2026-08-01 later 5 — `tests/test_stable_prefix_phase3.py`), and a
paired ablation (or B4-private measurement) justifying each default flip. Expected (unchanged):
hard-suite ~78-80% → mid-80s; latency 2-3× only on the triggered tail.

**Phase 4 — long-horizon context discipline (LAST; after 1-3 prove out).** Structured
mandatory-section compaction template (intent / files+paths / decisions / active goals / next steps)
+ probe-based compaction eval (recall/artifact/continuation/decision probes, not ROUGE) +
post-compaction constraint re-pin audit (the "governance decay" class: compaction silently erases
standing constraints) + deterministic pre-compaction cleanup (dedup file reads, purge resolved
errors). Proactive memory injection as a worker-node monitor, **append-only at the tail** — never
mutate earlier context (KV thrash otherwise). Turn-loop surgery: needs Phase-0 instruments and the
Phase-2/3 verifier to measure honestly.

**Cross-phase acceptance rule — AMENDED 2026-07-30 (operator decision, option c).** The synthetic
B4 battery ceilinged twice in one day (single-step 33/33, compositional 10/10 — §6): the agent has
outgrown short sandbox batteries, so §4F phases are judged OBSERVATIONALLY on live trajectories
over a ~2-week window (the earn-keep post-mortem's pre-approved "instrument changes" route).
Concrete watch set (all instruments already live): (1) verifier — periodic verify_bench re-runs
(NOT saturated; the one controlled instrument left) + live refute-escalation overturn rate (FP
proxy, `escalated_overturn`) + correction churn; (2) task outcomes — graded outcome-label pass-rate
trend (Tier-1 labels) + failed_retrievals arm; (3) learning-health telemetry (lesson utility,
hydration, activation counters). Phase-3 default flips (probe recalibration, BoN) proceed one at a
time as enable→watch→revert-if-worse sequential comparisons, never in bundles. HONEST CAVEAT:
observational evidence is weaker than the paired-ablation standard (no control arm) — treat trends
as directional, demand the controlled verify_bench for anything verifier-shaped, and PARK Phase 4
unless the observational picture is clearly positive. B4 infra (incl. the 10 comp tasks + live
pilot instrumentation + --only-ring) stays in tree as dormant, revivable for deep discovery-chains
(fork option a) opportunistically.

**WATCH STARTED 2026-07-30 ~16:00 (T0).** Prod restarted with logs reset; tuned verifier templates
confirmed LOADED in the live process (loader lines 15:57:58/15:58:07 + activation counters; note:
the claim path fires on evidence-shaped turns — CODE-exec turns ride _VERIFY_CODE_PROMPT and never
touch the tuned templates, which is why a code probe showed ⚠ 0-applies; not a defect).
T0 bundle: `ablation_out/watch-4f/t0/` (pre-ship + post-ship + probe-A/B verify_bench results,
learning-health snapshot, optim artifact sha256s). READING SCHEDULE: T+3d log spot-check
(escalated_overturn count, correction churn, false-refute complaints, BoN/probe still default-off);
T+7d verify_bench re-run vs t0 post-ship numbers + outcome-label trend; T+14d verdict per the
amended acceptance rule (incl. the FPR-regression watch: if live false-refute churn is up
noticeably, consider re-optimizing with the rebalanced clean-weighted trial mix FIRST).

**═══ NEXT STEPS — §4F, 2026-08-04. THIS BLOCK SUPERSEDES the 07-30 snapshot below ═══**

Ordered by expected value. Every item names what blocks it and how it gets verified; the 07-30
snapshot is kept underneath for its watch-caveat detail (a)-(g), which is still correct.

**0. WATCH THE DEPLOY BEFORE ADDING ANYTHING. (today → +1 day.)** A default-ON verifier behaviour
change (`_escalate_confirm`) went live at the 11:52:21 restart. Two changes in one window is the
exact confound that made the §4F watch unattributable the first time. The instrument:
```
jq -r '.route+"/"+.kind+"/"+.outcome' $GHOST_HOME/system/verifier/escalations.jsonl | sort | uniq -c
```
Absence of the file is EXPECTED (it is created on the first escalation after a boot). What to look
for: `claim/confirm` withheld rows accumulating materially faster than the ~4%-of-turns estimate,
which would mean the trigger is broader than measured and wants a second look before it becomes
load-bearing. `introspect action='learning'` now renders the same rates per (route, kind).

**0b. ⏳ TODO — RUN THE ESCALATION AUDIT once the ledger has filled. Time-gated, ~3-5 days from
2026-08-04. THIS CAN OUTRANK ITEM 1 — check it before committing to the macro work.**

```
GHOST_HOME=/Users/vasilis/Data/AI/Data PYTHONPATH=src \
  /Users/vasilis/Data/AI/.agent.venv/bin/python scripts/escalation_audit.py --limit 20
#   ... --json  → feed the cards to a judge instead of hand-scoring
```

*Why it exists.* The whole §4F reading of the 84% claim-refute overturn rate — "the cheap judge
false-alarms and the strong model corrects it" — is an ASSUMPTION, and req `03b96c28` is a live
counter-example: the cheap judge was RIGHT, the main model overturned it, and a fabricated `0` was
backfilled into the corpus as `passed`. **One case is not a rate.** This audit turns it into one.

*What to score.* Each card asks "was the cheap judge right?". The load-bearing field is
`tools_failed` — an overturn on a turn where every tool failed is where a fabrication becomes a
pass. The population to watch hardest is **partial** failure (141 live turns have SOME failed tool,
61 of them end `passed`): the shape rule shipped 2026-08-04 closes only the TOTAL-failure subset
(8 of 1491 turns), so partial failure remains pure model judgment with no floor under it.

*Supply.* ~8 escalations/day, and the ledger is created fresh on the first escalation after each
boot (it survives restarts; it is append-only with rotation at 4 MB). It held 1 row at the
14:37 restart. **Do not read a rate off fewer than ~20 decided rows** — that is the same
small-n trap that produced the "100% overturned" reading on n=4 (§4F T+3d) and the ±0.08 private
gate.

*Decision it feeds.* If overturns on tool-failed turns are a material fraction, refute escalation
is CORRUPTING outcome labels rather than repairing them, and fixing that outranks every optimizer
item below — the corpus has been absorbing false passes for as long as escalation has been on. If
they are rare, `03b96c28` was an anecdote, the current configuration stands, and this closes.

*Related loose ends: BOTH CLOSED 2026-08-04* — the shadowing kwarg is renamed
`unacked_total_failure`, and `f78c8b33` is repaired to `failed`/`structural failure` with a
corpus-wide sweep confirming 0 laundered records remain (details in the shape-rule §6 entry).
**A watch item they leave behind:** both post-deploy probe turns were refuted by the VERIFIER for
violating a constraint the environment made unsatisfiable ("reply with just the number", file
unreachable). n=2 and not acted on — but this audit is the natural place to notice if honest,
correct replies are being labelled FAILED at rate, which would be a verifier-side fix.

**1. BATCH / MULTI-PATH `file_system` CALL — highest value, UNBLOCKED, do this first.**
Re-measured 2026-08-04 (`scripts/tool_ontology_report.py`, non-overlapping step counts):
| sequence | occ | turns | steps removed | cohesion |
|---|---|---|---|---|
| `file_system ×3` | 595 | 78 | **500** | 0.69 |
| `file_system ×2` | 789 | 108 | 456 | 0.72 |
| `file_system ×4` | 477 | 58 | 471 | 0.69 |
Shared targets are real files (`model.py`, `train.py`, `index.html`), so cohesion 0.69-0.72 means
this is the SAME target touched repeatedly — a batch call is the right shape, not a speculative
merge.

⚠ **SCOPED 2026-08-04 with OPERATION-level data — "batch" is THREE macros, and this item's original
framing named the SMALLEST one.** The ontology report sees tool NAMES only; reading the actual
`operation` args off 1492 live trajectories (1138 ops inside runs of ≥2; run-length histogram
76×2, 39×3, 21×4, long tail, and one run of **144**) gives the split that decides the design:

| adjacent pair | count | same file |
|---|---|---|
| `read_chunked → read_chunked` | 181 | **85%** |
| `read → read` | 168 | 25% (75% DIFFERENT files) |
| `read → replace` | 95 | 88% |
| `replace → replace` | 66 | 87% |
| `replace → read` | 43 | 90% |

1. **Paging** (`read_chunked` ×2, 85% one file) is NOT batching — it is pagination, and the fix is
   a smarter/larger read (range list or auto-continue under a byte budget). **Separate item.**
2. **Multi-file read** (`read → read`, 75% different files) is the classic batch — a paths list.
3. **Edit cycle** (`read→replace` + `replace→replace` + `replace→read` = **204 pairs**, ~88% one
   file) is the BIGGEST, and it needs multi-replace in one call plus enough post-edit state
   returned that the trailing verify-read becomes unnecessary.

**Sharpest risk, must be handled explicitly:** a batch where 1 of 5 paths fails must not read as a
wholly-failed call to `outcome_heuristics.tool_call_failed` / `looks_like_tool_error` — the shape
rule shipped the same day keys on "did EVERY tool call fail", so a mislabelled batch result moves
outcome labels directly. **And the new parameter changes the advertised tool schema**, i.e. the
system prompt, so it MUST register in `CONTEXT_MUTATING_KEYS` or it silently contaminates the
Phase 2b fixture corpus that the optimizer replays verbatim (`GHOST_LLM_RECORD=1` is on).

**Also found while scoping (own item):** one recorded op name is corrupt —
`write\n<arg_key>content</arg_key>\n<arg_value>#!/usr/bin/env python3…` — an argument leaked into
the operation field (1 occurrence, the known replace-parser marker-leak class).

**✅ BUILT 2026-08-04 — `fs_batch` arm, STAGED not running (needs a restart).**
Suite **11157 passed / 14 skipped / 0 failures** (+38), verified independently.
- **(2) multi-file read** — `paths` on `operation='read'`, each entry optionally carrying an inline
  range (`train.py:120-180`, which also absorbs the 26% same-file `read→read`). Tolerant transport
  parsing (list / JSON array string / newline / comma) because the live XML dialect delivers every
  argument as a STRING.
- **(3) edit-cycle step-remover** — `post_edit_view`: a successful `replace` returns the changed
  lines as they NOW are on disk with line numbers, hooked inside `_write_replace_guarded` (the one
  write site every ladder rung passes through, so it cannot claim a change that was rolled back).
  The "several replacements per call" half ALREADY EXISTED (concatenated `<<<< SEARCH` envelopes) —
  advertised in the treatment prose rather than reimplemented.
- **Deliberately NOT built, with numbers:** cross-file multi-edit (only ~8 of 204 edit-cycle pairs
  cross a file, not worth touching the SEARCH/REPLACE parser and its 3 fallback rungs);
  `read→replace` collapse (95 pairs — look-before-you-leap, and semantically incoherent to batch:
  you cannot write a SEARCH block for text you have not read); batching any MUTATING op (only
  `read` fans out). Noted for later: `search→read`/`search→read_chunked` (31 pairs, ~84% same file)
  would fall to the same "return more state" trick applied to `search`.

**Partial-failure rule (the sharpest risk, handled):** THREE live classifiers decide "did this call
fail" — `agent._res_is_error`, `outcome_heuristics._looks_like_tool_error` (first 120 chars), and
`composed_skills._step_result_ok`. Shipped rule: **≥1 path read → PARTIAL and must not look failed
to any of the three; 0 paths read → `Error:`-prefixed.** The guard is envelope LENGTH, because
per-path bodies legitimately contain `Error: 'c.py' not found` and the header must cover the
120-char sniff window. Mutation-checked at the classifiers' own granularity. Multi-edit atomicity:
best-effort per block, **atomic per file**; bounding delegated to the existing `ReadBudget`, with
one new 12-path fan-out cap reported by name, never silent.

**Acceptance — simulated upper bound** (`scripts/tool_ontology_report.py --simulate-fs-batch`,
reproduced independently). The literal before/after needs live uptake:
| n-gram | occ | steps | cohesion |
|---|---|---|---|
| `file_system ×2` | 789 → **585** | 456 → **349** | 0.72 → 0.83 |
| `file_system ×3` | 595 → **414** | 500 → **364** | 0.69 → 0.81 |
| `file_system ×4` | 477 → **319** | 471 → **330** | 0.69 → 0.82 |
file_system calls 1138 → 934 (−17.9%); all tool calls −5.7%; 51 turns affected (5.8%). Rising
cohesion is the CORRECT residual — what remains is paging and same-file edit cycles, both
deliberately out of scope.

**Targeting (verified independently, and it is the point).** Split of 1340 user turns on "has a
file_system run of ≥2": **n=108, mean depth 20.6, failure 47.2%** vs **n=1232, depth 4.0, failure
8.8%** — the targeted population fails at **5.4× the base rate**. The 51 collapsed turns are a
tighter subset still (depth 24.4, failure 58.3%). ⚠ **This is a TARGETING result, not an effect** —
the same discipline §4I applied to the risk governor. Deep turns carry many file_system calls
BECAUSE they are hard; removing calls may shorten them without making them succeed. Whether it
helps is the `fs_batch` arm's job to answer.

**✅ LIVE since the 2026-08-04 16:23 restart at traffic 1.0** (operator decision). Boot line reads
`experiments — 2 live: risk_steer traffic=1; fs_batch traffic=1`. No `experiments.json` exists, so
`DEFAULT_SPECS` is authoritative.

**═══ SEQUENCING DECISION — fs_batch vs Phase 2b (operator, 2026-08-04) ═══**

*The conflict.* `CONTEXT_MUTATING_KEYS` registration was mandatory (the schema change alters the
system prompt), and it excludes the ENTIRE treatment arm from new Phase 2b fixtures. `traffic` is
the fraction ENROLLED and arms split within it, so at **1.0** the split is 50% control / 50%
treatment ⇒ **~50% of new fixture supply is excluded**; at 0.5 it would be 25/25/50-unenrolled
⇒ ~25% excluded but the experiment verdict takes TWICE as long.

*Decision: run at 1.0, and re-mine Phase 2b LATER rather than resetting anything now.* Reasons,
all measured:
- Phase 2b sits at **66/~250 positives** and cannot clear before ~08-17 regardless, so `fs_batch`
  resolves well ahead of it — the supply this costs is supply that was not going to be spent yet.
- Phase 2b's own ceiling check measured **0.772 incumbent fidelity**: its upside is prose
  refinement. `fs_batch` targets the population that fails at **47.2% against an 8.8% base rate**.
  Holding a depth intervention to protect a prose optimizer is the worse trade.
  ⚠ **RE-MEASURED 2026-08-05 (see §6): 0.821 (55/67, 0 unreplayable).** The 0.772 was
  `fidelity_runner`, which scores 5 replay-plumbing failures as wrong; the toolbox-only figure that
  day was 0.846. The *sequencing* conclusion above still holds — a ~0.82 incumbent is still a prose
  ceiling — but do not quote 0.772 as a measurement of the toolbox. **And the clustering premise
  below is not reproducible:** 12 misses on 12 distinct pairs, none twice.
- **The real invalidation is AHEAD, not behind.** `file_system` is **85 of ~190 fixtures** — the
  dominant tool. If `fs_batch` wins and becomes default, every file_system fixture was mined
  against a schema that no longer exists. **So when fs_batch resolves, bump
  `DEFAULT_ERA_CUTOFF_LOCAL` and re-mine** (one line; the era mechanism already exists and was
  built for exactly this on 07-31). That is a targeted reset WITH a trigger, not a reset now.

*Considered and REJECTED: a wholesale "reset the invalidated GEPA data and start over".* Almost
everything a reset would clear was rebuilt the same day — the mined verify_bench pool (86 cases),
the gate baseline 0.766 measured with both escalation directions live, the t1 bundle, and the
ontology re-run. `planning.decompose` and t0 were already de-recorded. A reset now would redo hours-
old work and discard 66 hard-won positives for no benefit.

*Also considered and REJECTED: bumping the calibration epoch.* `CURRENT_EPOCH` stays
`2026-07-27.graded`. `calibration.py`'s own contract says a label-scheme change MUST start a new
epoch, and the shape rule technically is one — **but it is eligible on 9 of 1499 turns (0.60%)**,
while a bump discards the whole current corpus (n=635) and silences the fit until it re-accumulates.
That rule exists for changes like 07-27, which moved the base rate 0.955 → 0.855. **Recorded so a
future session does not read this as an oversight** — it is a decision, with the number attached.

**The corrupt op name is 17 occurrences, not 1** — 15 of them `file_system`, in TWO dialects: the
native-template `<arg_key>` shape and the equals-dialect (`<parameter=path>`) leaking into an
`operation` or `path` VALUE (e.g. `'operation': 'read_chunked>\n<parameter=path>\nindex.html'`).
Format 1's lookahead should stop at `<parameter=`; the suspected amplifier is the **Format-5b
bounds-aware repair pass, which replaces a value whenever the repaired body is merely LONGER**.
An `agent.py` parser defect — **its own item, not fixed here.** The new `paths` parameter adds no
exposure (one more parameter, exactly like `path`) and fails legibly: a corrupted JSON array fails
`json.loads` and is deliberately NOT comma-split, since that would manufacture filenames out of
JSON punctuation. Test feeds it a verbatim live-corruption shape.

**Two defects caught in its own code by self-review, before the suite:** `resolve_batch_entry` was
applied to a plain `path` in the treatment arm, so a real file named `logs/2026-08-04:12` would
have been silently retargeted (tolerant-parser false positive — the recurring class);
and `post_edit_view` used quadratic difflib on an otherwise linear path (now O(n) above 20k lines).
**Cross-surface guards tripped by ONE new log icon:** `app.js` ICON_CLASS coverage, the uConsole
`turnstatus.py` hand-synced copy, and the `app.js`/`matrix_graph.js` cache-bust lockstep — all
synced (`?v=8.9 → 9.0`). ⚠ **The uConsole client still needs a deploy to pick up `turnstatus.py`.** **Why it outranks everything else:** depth is the strongest failure predictor measured here
(17.8% at step 1 → 60.6% at step 12) and the risk governor CANNOT fire before step 6 by
construction — both its terms need a turn to have already happened. Every other lever steers a turn
that is already deep; this one makes turns shorter. It improves failure rate and latency with one
edit. **Ship rules:** as a `core/experiments.py` arm, never as a default; register a compliance key
in `TRIGGER_KEYS`; and if it mutates prompt context, in `CONTEXT_MUTATING_KEYS` — otherwise it
silently contaminates the Phase 2b fixture corpus (the optimizer replays payloads verbatim).
**Verification is direct:** re-run the ontology report; those n-grams must collapse. If they do not,
the macro did not land, whatever the latency says.

**2. PHASE 2b TOOL-DESCRIPTION RUN — blocked on supply until ~2026-08-17, and now ALSO gated on
`fs_batch` resolving.** ⚠ **Do not run 2b before the `fs_batch` verdict.** `file_system` is 85 of
~190 fixtures; if `fs_batch` wins and becomes default, every one of them was mined against a schema
that no longer exists. **When fs_batch resolves: bump `DEFAULT_ERA_CUTOFF_LOCAL` and re-mine**
(one line — the era mechanism exists for exactly this). Until then the treatment arm is excluded
from fixtures by design, so supply accrues at ~half rate — deliberate, priced, see the
sequencing decision in §6. Gate is ~250 positives
(see the Phase 2b supply block above for why it is not 200). Check without starting a run — the
miner now prints the runner's own resolution verdict:
```
GHOST_HOME=… PYTHONPATH=src python -m scripts.mine_tool_fixtures
```
**At gate-clear, in this order:** full-set incumbent fidelity check FIRST (0.772 on 08-03 = real
headroom, so this is a formality unless it moved) → GEPA adapter over the fixtures → private gate →
promote → deploy by restart → **confirm the activation counter reads `applied N`** (an artifact on
disk with 0 applies is a dead read-site, the defect that counter exists for) → **then flip
`GHOST_LLM_RECORD` off in the launcher and archive/delete the day-files.** Standing cost of every
day this slips: unredacted recordings on local disk.

**3. NEXT VERIFIER ROUND — unblocked, deliberately NOT next.** Target: beat
`private_incumbent_balanced = 0.766` under `--escalate gate`. **Direction is VERDICT-TIER ROUTING
(architecture), NOT another prompt round** — two prompt strategies are cleanly refuted by the gate
(rebalanced −0.131, compression-capped −0.119) and the diagnosis is capacity-bound rule-following:
the E4B judge cannot APPLY rules it already carries in a 5.4 KB prompt. More text is the thing
measured not to work, twice. Reasons to deprioritise: the live FP churn is escalation-bounded
(~8/day, latency-only), and a round costs ~75 min of the main slot. **Before spending that:**
re-record the baseline so `route_health` and `confirm_eligible` are present (`--incumbent-only`) —
without them, cheap-leg fall-through cannot be ruled out and "0 confirms withheld" is ambiguous.
**Known cost to fix or accept first:** the CONFIRM direction cannot move the gate metric
(`_trial_score` is verdict-only; a withheld confirm changes only confidence), so `--escalate gate`
spends one main-model call per eligible trial for zero gate signal.

**3b. ❌ NOT A BUG — CLOSED 2026-08-04 WITHOUT A CODE CHANGE. The leak was already fixed on
07-31; all 17 occurrences are HISTORICAL.** Dated against the `QWEN_TOOL_PROMPT_NATIVE` split
(2026-07-31 ~18:54): **0 of 17 occurred after it**, and **161 trajectories have been recorded since
with zero corruption**. Per-day: 07-08 ×2, 07-14, 07-17 ×3, 07-18, 07-20, 07-24, 07-25 ×2, 07-27 ×2,
07-31 ×4 (all pre-fix). The memory note "repair fires ≈0 now, each one is news" was correct.
⚠ **This item was WRONG as I first wrote it** ("17 live occurrences … sits in the hot tool path") —
the subagent reported a count without dating it and I propagated that framing. **Dating a defect
against the fix that was supposed to close it is the check that was missing**, and it is the same
lesson as [[measure-the-mechanism]]: a corpus-wide count is not a live rate.
*Two real residuals, both small:* (a) the corrupt records are ALREADY excluded from Phase 2b
fixtures (era cutoff 19:15 > the last occurrence at ~18:54) — no action; (b) the ontology analysis
does NOT era-filter, so **8 corrupt `file_system` operation values of 1138 (0.70%)** still enter the
macro counts. Immaterial to the fs_batch conclusion but it is a real instrument impurity.
*✅ BOTH RESIDUALS CLOSED 2026-08-04 (operator: "do both").* New `utils/leaked_framing.py` — a
corpus DIAGNOSTIC, deliberately **not** the repair predicate (`agent._value_has_leaked_framing`
decides whether to TRUNCATE a live value and must stay strict; counting is not truncating). The
strict predicate matches only 11 of the 17 known corruptions: six have a clean prefix then a
sibling `<parameter=` with **no preceding close token**, and one uses the `<arg_key>`/`<arg_value>`
dialect whose tokens the repair regexes do not list at all.
- **Position is the discriminator**, measured against the real shapes: framing at the START of a
  value, or immediately after a newline, or appearing twice. Audited both ways — all 6 live shapes
  caught, and `"the XML dialect uses <parameter=path> style tags"` / `"def f(): return
  '</tool_call>'"` correctly NOT flagged. Without that rule the diagnostic fires on any docstring
  about the dialect, which is broken in the other direction.
- **Ontology mining now excludes them** — 16 of 3579 calls (0.45%). Their `operation` and target
  are fiction, so they polluted both the n-gram counts and the cohesion denominator; removing them
  moved the 2-gram 789→776 and the 3-gram 595→584 (~1.5%). Printed as a `corpus purity` header, not
  applied silently.
- **Recurrence watch in learning-health**, keyed on the **newest occurrence, not the count** — the
  corpus is append-only so the 16 never go away. Renders "all historical — no action" today, and
  "⚠ REGRESSION" the moment a post-fix timestamp appears. **This is the listener "each one is news"
  assumed existed.**
- 26 tests (`tests/test_leaked_framing.py`), docs in `docs/tools/introspect.html`.
*Test-writing gotcha worth keeping:* the collector globs `session-*.jsonl` under a `YYYY-MM-DD`
partition — a fixture written to any other filename is SILENTLY invisible, and the test then passes
for the wrong reason (mine failed loudly instead, which is why it was caught).

**(original framing, kept so the correction is legible)** ~~TOOL-CALL PARSER: arguments leaking
into `operation`/`path` VALUES. 17 live occurrences, 15 of them `file_system`. UNBLOCKED, small,
and it sits in the hot tool path.~~
Two dialects: the native-template `<arg_key>` shape
(`'operation': 'write\n<arg_key>content</arg_key>\n<arg_value>#!/usr/bin/env python3…'`) and the
equals-dialect leaking into a value (`'operation': 'read_chunked>\n<parameter=path>\nindex.html'`,
`'path': '</parameter>\n</function>\n</tool_call>\n<tool_call>…'`). **Format 1's lookahead should
stop at `<parameter=`; the suspected amplifier is the Format-5b bounds-aware repair pass, which
replaces a value whenever the repaired body is merely LONGER.** Verify that suspicion before
changing anything — it is a hypothesis, not a diagnosis.
*Why it matters more than 17 sounds:* this is the same marker-leak class that once wrote `====`
into real files ([[replace-parser-marker-leak]]), and a corrupted `path` value is a tool call
pointed at the wrong file. Also: every one of these is a MIS-RECORDED tool call, so the §4F Phase 2b
fixture corpus and the ontology analysis both count them as their corrupt string.
*Not made worse by `fs_batch`:* `paths` is one more parameter exactly like `path`, and it fails
legibly (a corrupted JSON array fails `json.loads` and is deliberately NOT comma-split, which would
manufacture filenames out of JSON punctuation). There is a test feeding it a verbatim live
corruption shape.

**4. OPERATOR DECISIONS — all three measured, none urgent, none mine to make.**
(a) **Distill vs prune disagree by construction** — minted lessons score 0.3633 / 0.7943 against a
live cutoff of 1.0716. Harmless while `GHOST_SKILL_PRUNE` is OFF; a landmine the moment it flips.
(b) **The complexity router is measurably good and 100% inert** — CV lift +0.138 over 837 live
decisions, both consumers dark, `use_planning` not even an argparse argument. Wire it or retire it;
§4I Phase 2's backtest needs ~a week of stamps first (coverage 0.8% today).
(c) **Nova's `-np`** — all 9 `ReadTimeout` events sit exactly on the 12 s route ceiling. More slots
is the fix; raising the timeout again moves the wall.

**5. DO NOT START.** Another verifier PROMPT round (refuted twice — see item 3). Reviving
`planning.decompose` (its promotion is no longer a measured win; parked item 6). Phase 4 context
discipline (still gated on the observational verdict). Any Phase 2b work before the supply gate.

⚠ **The T+14d observational verdict (~08-13) needs a new method.** Its reference bundle
`ablation_out/watch-4f/t0/` is superseded by `t1-20260804/` — case pool, escalation arm, GEPA metric
and calibration key scoping all changed underneath it, so a t0 diff reports INSTRUMENT changes as
behaviour. Judge against t1, and note that `core/experiments.py` now offers what the whole
observational design was a workaround for: a real control arm.

**═══ WHAT REMAINS — §4F snapshot, 2026-07-30 end-of-day (superseded above; kept for caveats a-g) ═══**

**Two clocks running (both converge ~2026-08-02 → 08-13):**
1. **§4F observational watch** (T0 = 2026-07-30 ~16:00, bundle `ablation_out/watch-4f/t0/`):
   T+3d (~08-02) log spot-check — `escalated_overturn` count, correction churn, false-refute
   complaints; T+7d (~08-06) verify_bench re-run vs t0 + graded outcome-label trend; T+14d (~08-13)
   VERDICT per the amended acceptance rule → **decides Phase 4 (parked unless clearly positive)**.
   Ask any session for "the 4F watch reading".

   **T+3d READING — CORRECTED after reconciling with the 08-01 T+2d reading (see below):**
   my window label was WRONG (the log rotated during the 07-31 deploy churn — the "since T0"
   numbers below actually cover ~08-02→08-03 only), and the T+2d reading's facial-validity method
   overturned my first-pass diagnosis. THE REAL SIGNAL IS THE SHIFT: T+2d (07-31→08-01 segment)
   measured ~130 cheap refutes with 49% overturned and 4/4 sampled SURVIVING refutes true
   positives (healthy churn); my segment (~08-02→08-03) has 74 overturns and ZERO surviving
   refutes — overturn rate went 49% → 100% right around the 08-01-evening claim-fairness deploy.
   Facial check on MY segment's overturns: the cheap judge is refuting subjective glosses
   ("beautiful hot Sunday evening"), rc-vs-stable version classification, and instrumentation
   echoes — the EXACT hand-pinned false-alarm rules the baseline adjudicate carried and GEPA's
   rewrite SHED (the bench's 13 clean cases never exercised those patterns — Goodhart on the
   under-measured class). Repairs: 0 since current boot (the 07-31 dual-dialect fix HOLDS; my 70
   count was pre-fix residue). Calibration-map warnings: covered by the operator's 08-02
   calibration-epochs entry (shipped, not yet deployed). RESPONSE (operator green-lit): balanced
   re-optimization launched 2026-08-03 — 8 FP-trap clean cases APPENDED to the bench (subjective
   gloss, derived arithmetic/count/units, rc-vs-stable, paraphrase, instrumentation echo, extra
   detail — the shed rules, now encoded in the METRIC), class-weighted scores (refute_weight
   0.263), seeded FROM the live templates, ship-gate = BALANCED macro-average vs the LIVE
   incumbent on 30 private trials.
   **REBALANCED RE-OPTIMIZATION RESULT (2026-08-03 evening): REJECTED by the private gate** —
   candidate 0.708 vs live incumbent 0.840 balanced on 30 never-seen trials (public-valset winner
   overfit; the gate's third correct rejection; candidates kept as .rejected). **AND the diagnosis
   is REVISED: reading the LIVE adjudicate disproved rule-shedding** — every July FP rule is
   present (GEPA even ADDED a numerical-precision rule). Actual mechanism: **capacity-bound
   rule-following** — the E4B judge fails to APPLY rules it carries in a 5.4KB prompt + long
   evidence, while the 35B with the SAME prompt applies them on escalation (hence 100% overturns).
   More rule text cannot fix this (today's run is the demonstration). NEXT DIRECTION (needs
   green-light, proposed for T+7d): COMPRESSION-constrained optimization — hard length cap on
   adjudicate candidates (~1500-2000 chars) so the small judge can actually follow what it reads;
   alternatively verdict-tier routing. FP-trap cases + balanced gate stay (they made this
   measurable). Live churn (~18 overturns/day, latency-only) accepted until then. Memory
   `optimizer-sheds-pinned-rules` corrected same evening.
   **(original first-pass reading below, kept for the record; window label corrected above)**
   **The pre-registered FPR clause FIRES.** 74 distinct escalation-overturn events in the window
   (~18/day), **zero standing refutes** — every cheap-judge refute was a false alarm the main-model
   escalation overturned (live-demonstrated: a correct "Linux 7.1.5" probe answer was refuted, then
   overturned). Damage stayed bounded exactly as designed (latency-only: ~74 main-slot escalation
   calls), but the tuned adjudicate is generating pure-noise refutes on live traffic → **next
   verifier optimization must run with the rebalanced clean-weighted trial mix, and should run
   SOON rather than waiting for T+14d** (operator green-light pending). Activation healthy:
   templates load per process, applied>0 after claim-shaped turns; overnight 0/0 was traffic shape
   — most live verdicts ride the CODE-auditor path, which is NOT GEPA-tuned (scope note: tuning
   _VERIFY_CODE_PROMPT may matter more for this traffic mix than another claim-template round).
   Phase-3 switches confirmed OFF. **Anomalies for follow-up:** (a) native tool_call repair fired
   70× (~17/day) vs the "≈0, each one is news" baseline — regression, model-churn template drift
   suspected, wants a dedicated look; (b) 52 circuit-breaker opens for 192.168.0.20:8088 in one
   ~30-min afternoon burst (address absent from current args — transient/operator-experiment
   suspected; watch recurrence); (c) restart cadence: 27 processes reached the claim path in
   ~4 days — confirm with operator whether restarts were manual; (d) 116 "calibration REJECTED the
   probability map" + 42 PRM serve-inert warnings — fold into the next log-audit session.
   **Phase 2b supply: 636 structured tool-call fixtures** (file_system 290, execute 116, browser
   114, …, web_search 16) — minable now; rare-tool coverage improves with a few more days. **⚠ Reading caveats (audited 2026-08-01):**
   (a) the honest-failure rule (07-31 ~19:14) is a LABEL-SEMANTICS step change — outcome-label
   trends must segment at that boundary or the relabeling masquerades as improvement; (b) refute
   greps must match ALL spellings ("verifier gate — REFUTED", "LATE REFUTED", "refuted (late)") —
   naive "verifier — REFUTED" counts ZERO; (c) the 17-boot day (07-31) was operator deploy
   cycles, NOT instability — uptime metrics should exclude it; (d) in-process activation counters
   reset per boot — use the loader log lines for cross-boot activation evidence; (e) ATTRIBUTION
   confound (added 2026-08-01): the native dual-dialect fix (07-31, stacked-call corruption
   18 fires/2.5h → ≈0), the honest-failure relabel, AND the 08-01 later-9 claim-fairness ship
   (pack_claim + strip_system_notes + constraint lifecycle — directly reduces false-refute
   incidence; verify_bench provably unaffected, but LIVE refute/overturn counters are not) all
   improve live outcomes mid-window from NON-verifier causes — the T+14d observational verdict can report "the stack improved" but
   cannot attribute to the verifier ship alone; the CONTROLLED verify_bench re-runs (T+7d, same
   judge + templates, artifact hashes pinned in t0/) are the only verifier-specific evidence and
   carry the §4F verdict weight accordingly; (f) LOG COVERAGE (verified 2026-08-01): the live log
   was reset again at a 07-31 ~21:02 deploy boot — log-based counters cover ONLY from there
   (conveniently ≈ the post-honest-failure segment; rule landed 19:14). The first ~29 h of the
   window live in the DURABLE stores instead (trajectories + corrections.jsonl at the trajectory
   root + autonomous_activity ledger). Overturn counter line shape confirmed greppable:
   "Verifier escalation OVERTURNED a cheap-judge refute" (dedupe the GhostAgent/GhostStream twin
   emission — one event logs twice). **(g) T0's calibration block is NOT comparable across the
   2026-08-02 epoch fix** (§6 that date). `ablation_out/watch-4f/t0/learning_health_t0.json` snapshots
   `learning_health.calibration`, and several of its keys are now epoch-scoped rather than whole-file,
   plus two verdict rules changed. A naive T+7d/T+14d diff will show step changes that are INSTRUMENT
   changes, not behaviour — same trap as caveat (a). Specifically: `entropy_observed_pct` 12.3 → ~77
   (denominator 1709 → 541); `entropy_observed_samples`, `outcome_pos`/`outcome_neg` (now split at 0.5,
   not exact 1.0/0.0), `label_*` and `feature_health` all re-scoped; `feature_health` verdicts now use
   the fit's 2.5σ gate, so `live_features` drops 2/4 → 1/4 and the T0 line
   `competence_component: separation 0.0023 [dead]` was itself measured ACROSS the label-scheme change.
   `samples_on_disk` was deliberately left whole-file and stays comparable. **The calibration/confidence
   stack is otherwise independent of §4F** — `optim/` never reads it, `grade_turn_outcome` is unchanged
   so the graded-outcome-label trend is bit-identical, and Phase 3 item 4(a)'s "threshold-aware
   calibration" is the verifier logit-probe blend, not this. For a calibration reading at T+7d/T+14d,
   re-baseline from a fresh post-fix `introspect action='learning'` rather than diffing t0.
2. **Phase 2b fixture supply** (`GHOST_LLM_RECORD=1` in the launcher since 07-30 16:06 —
   UNREDACTED, local-only): **supply was structurally broken until 07-31 ~13:40** — the main tool
   loop STREAMS and the recorder had no streaming hook (the §4-parked "dev feature" turned out
   load-bearing: 21 h of heavy traffic = 7 fixtures). IMPLEMENTED + DEPLOYED 07-31: stream-side
   recording in `_do_stream_chat_completion` (delta accumulation incl. indexed `delta.tool_calls`
   fragments — native-tools streams carry the parsed call THERE with empty content; reassembled
   OpenAI-shaped record on clean completion only, kind=`chat_completion_stream`; zero cost when
   recording off; `tests/test_stream_recording.py`, 10 tests). Live-verified: probe turn recorded
   tool calls with names+arguments. Miner ground truth = `message.tool_calls` (structured — the
   earlier content-parsing contract note is OBSOLETE for streamed records).
   **UPDATE 2026-08-01 (later 5): the miner (`optim/tool_fixtures.py` +
   `scripts/mine_tool_fixtures.py`) and the registry read-site
   (`tool_description.<tool>.json` artifacts via `_apply_tuned_descriptions`) are BUILT, tested,
   and live-smoke-tested (111 fixtures mined to scratchpad; supply gate says wait).** Remaining:
   after ~2-3 days of supply → run the miner for real, write the GEPA adapter (rehydrate payloads
   via fixture `source` pointers; fresh RequestState per candidate — the tool-defs/XML caches key
   on tool NAMES), gepa run, private gate, deploy via restart — **then flip recording OFF +
   archive/delete day-files**.
   **UPDATE 2026-08-03: the RUNNER is BUILT** (`scripts/optimize_tool_descriptions.py` — temp-0
   replay of recorded contexts against MAIN with candidate descriptions swapped copy-on-write;
   positives-only scoring; tier split honored; per-tool + aggregate validators reused; --smoke
   mode; supply gate = 200 POSITIVES). Smoke findings: (a) FIXED a replay bug — recorder ordinals
   are PER-SESSION, so the record lookup must match (session_id, ordinal), not ordinal alone
   (7/12 private fixtures read as unreplayable before the fix); (b) supply today = 57 positives
   (~28/day → gate clears ~08-08); (c) **⚠ CEILING RISK, B4-shaped: incumbent fidelity 1.000
   (12/12 private at temp 0)** — positives are by construction turns the model already chose
   well; the metric defends against regressions but may carry no improvement signal. RULE AT
   GATE-CLEAR: run the full-set incumbent fidelity check FIRST; if ≥0.95, CLOSE 2b as "no
   measurable problem" (the B4 precedent) — and flip recording OFF either way.
   **EARLY CEILING CHECK RUN 2026-08-03 evening (operator-approved shortcut): fidelity 0.772
   (44/57, 0 unreplayable) — 2b STAYS ALIVE, real headroom.** The 12/12 smoke was small-n luck.
   Miss pairs name the fixable confusions: browser↔file_system, execute→file_system,
   manage_projects over-selected (wrong wins + introspect/web_search misrouted into it), 3
   no-tool stalls, workspace/notify_operator→system_utility. Recording stays ON to the 200-positive
   gate (~08-08); the optimization runs there with genuine signal.
   **Compression-constrained verifier round LAUNCHED same evening** (operator green-light):
   caps 1200/2000 chars on MUTATED enumerate/adjudicate candidates (seed exempt as reference;
   over-cap candidates zero-scored with compress-feedback; ship condition enforces fits_caps) —
   targets capacity-bound rule-following on the E4B judge. Same balanced private gate vs the
   0.840 incumbent.
   **RESULT (2026-08-04 ~19:09): REJECTED — candidate 0.637 vs incumbent 0.756 balanced
   (−0.119, caps satisfied).** Two run-craft defects found+fixed en route: (a) run 3a burned 37
   iterations on SILENTLY zeroed over-cap candidates — the cap lived only in post-hoc scoring,
   invisible to gepa's reflector which composes from the PARENT's traces; fix = constraint
   embedded in `reflection_prompt_template` (where proposals are born) + loud cap-guard prints
   (operator's raw-stream observation caught it, not my grep summaries); (b) monitoring rebuilt
   as a StatusBoard (`<run-dir>/status.txt|json`, atomic, adapter+gepa-logger fed — phases,
   accept/reject/cap counters, verdict line); runs now resume via --run-dir. ALSO MEASURED:
   private-gate noise ±~0.08 across sessions on the 4-case private tier (incumbent re-read 0.840
   → 0.756, same templates) — **grow the private case pool before any further verifier prompt
   round.** VERDICT MEANING: two strategies now cleanly refuted by the gate (uncapped rebalanced
   −0.131, capped compressed −0.119); the 07-30 incumbent stays the best measured configuration;
   compression to ≤2 KB costs real bench accuracy on this judge — the remaining lever from the
   capacity diagnosis is VERDICT-TIER ROUTING (architecture, not prompt text; T+7d discussion),
   plus the standing FP churn stays escalation-bounded (~18/day, latency-only).

   **SUPPLY RE-MEASURED 2026-08-04 (later) — the gate is ~250 POSITIVES and lands ~2026-08-17, not
   200 by 08-08. Two independent reasons, both measured:**
   (a) The tier is hashed per **request** while one request emits 1-40 fixtures, so `--private-pct
   30` realises **20%** on positives (13 of 65 live). `--min-delta 0.02` needs 50 PRIVATE positives
   ⇒ **~250 positives**. The runner now REFUSES to start below that resolution (it had no
   resolution guard at all — the only one of the three runners without one).
   (b) The rate collapsed: joined real choice records **161/day (08-01) → 11 / 13 / 7**; positives
   **28 → 10 / 11 / 0**. Traffic did NOT drop (2200-3000 records/day) — the positive YIELD did.
   **Flag collision closed:** `mine_tool_fixtures --min-fixtures` counted ALL fixtures while
   `optimize_tool_descriptions --min-fixtures` counts POSITIVES, same default — the miner would
   have declared "ready" and ATOMICALLY OVERWRITTEN the live pool at ~71 positives. `--min-positives`
   added; the miner now prints the runner's resolution verdict before a run is started.
   **No self-play contamination:** of 219 unjoined records, **215 are self-play turns** with no
   trajectory to join and only **4** are real losses. The exclusion is incidental, not deliberate —
   know that before "fixing" it. **Ceiling is not the blocker:** fidelity 0.772 (44/57) is real
   headroom. ⚠ `GHOST_LLM_RECORD=1` stays exported the whole time and the day-files are UNREDACTED;
   every day the gate slips is another day of that.

**Phase 2b+ — tool ONTOLOGY analysis (2026-08-05, §6).** `optim/tool_ontology.py` +
`scripts/tool_ontology_report.py` measure the two structural questions description prose cannot
answer: which confusion pairs are BOUNDARY problems (bidirectional, evidence-tiered by an exact
binomial symmetry test) versus wording problems (one-way), and which recurring consecutive
tool-sequences are macro candidates. Read-only; promotion stays operator-gated. `--confusion-out`
on the 2b runner persists the misses the 08-03 ceiling check discarded. **First run: the
`file_system` batch gap dominates** (789 pair-occurrences over 108 turns, cohesion 0.72).
**Fixture isolation:** turns whose prompt context a live A/B treatment mutated are now excluded
from mined fixtures by default and COUNTED (`experiment_context_excluded`) — the optimizer replays
payloads verbatim, so a steered turn would tune descriptions against a context only one arm sees.

**Queued behind evidence (do NOT start early):**
3. ~~Next verifier optimization round MUST use a rebalanced trial mix first~~ ✅ **DONE and
   REFUTED — this item is CLOSED.** The rebalanced mix ran 2026-08-03 (candidate 0.708 vs incumbent
   0.840, −0.131) and the compression-capped round ran 2026-08-04 (0.637 vs 0.756, −0.119). Both
   rejected by the private gate. The rebalanced trial mix and the FP-trap cases STAY (they made the
   failure measurable); what is closed is the idea that another PROMPT round is the next move. See
   NEXT STEPS item 3: the direction is verdict-tier routing, and the gate baseline is now **0.766**
   on a tier that resolves 0.0093. ⚠ The "~5:1" ratio in the original text was measured FALSE
   (2026-08-04): it is **3.41:1** in the T0 bundle it cites, 3.06:1 today.
4. Phase 3 default flips, one at a time, enable→watch→revert: (a) probe needs a LIGHTER blend or
   threshold-aware calibration first (as-built it neuters actionable TPR: 0.347); (b) BoN
   (GHOST_TTS_ADAPTIVE_BON) after the probe question settles. Prereq for any flip: the
   stable-prefix-hash regression test (§4F Phase 3 TODO).
5. Phase 4 (context discipline): PARKED until the T+14d verdict.

**Parked/dormant (documented exit conditions):**
6. `planning.decompose` tuned artifact — consumer dark (no `--use-planning`). ⚠ **AND its promotion
   is NO LONGER A MEASURED WIN** (re-scored 2026-08-04 later: recall 0.429→0.857 reproduces the
   original promotion, token F1 **0.500→0.071** rejects it; median output 111 distinct tokens
   against a 32-token gold). It was promoted by the recall-only metric that made verbosity optimal.
   Keep the file, de-record the claim, do NOT revive on a paired ablation alone — re-promote only
   against a bench that grades plan QUALITY, since neither metric does. That signature now also
   trains on 96 examples that are 100% reflection-sourced. tool_selection.pick /
   reflection.critique — need signature-specific example extractors (generic trainset fits only
   planning).
7. B4 synthetic battery — dormant after double saturation (33/33 single-step, 10/10 comp);
   revivable via fork option (a) deep discovery-chains (5-8 stages, spec-in-previous-output,
   timeout > 300 s). The 10 comp tasks + live pilot instrumentation + `--only-ring` stay in tree.
8. ~~Minor: two `test_thinking_loop_guards` color tests fail under FORCE_COLOR-exporting shells~~
   ✅ RESOLVED 2026-08-01 (`_no_color_env_leak` autouse fixture pins `_USE_COLOR` + the
   import-time-baked ANSI constants; verified green under FORCE_COLOR=3).

**═══════════════════════════════════════════════**

### 4E. Negative-label supply — Tiers 1-3 ✅ DONE (Tier 3 2026-08-01), Tier 4 ⛔ HOLD

**Tiers 1-3 are DONE** (§6 "later 9"/"later 10" 2026-07-27; Tier 3 §6 2026-08-01 "later 5"). Tier 4 is
on HOLD. Measured supply per tier — do this BEFORE designing any further detector, it has already killed
two of my own designs:
| tier | measured supply | status |
|---|---|---|
| 1 graded label | every turn | ✅ done |
| 2 failure reports | 20/246 turns (8.1%), ~1/day | ✅ done (redesigned from evidence) |
| 3 reopened work | 7 project_reopened events / 20 days | ✅ done 2026-08-01 (marginal by design) |
| 4 generated probes | n/a — instrument failed before | ⛔ hold |

**Why any of this:** the label is ~96% one class. Two distinct problems hide in that — (a) too few
negatives (49 of 1226, no statistical power) and (b) the label measured "did anything visibly break"
rather than "was the answer good". Tier 1 fixed (b). Tiers 2-3 attack (a) with signals that ALREADY OCCUR
but are discarded.

- **✅ TIER 2 — DONE 2026-07-27 (later 10), but NOT as designed here.** The original plan (loosen
  phrase+rephrase to phrase-only) was **killed by measurement**: scanning 246 eligible session triples, the
  correction classifier fires on **0**, while 20 (8.1%) report broken work as a pasted traceback or "it still
  doesn't work" — neither of which contains a phrase to loosen. Shipped `classify_failure_report()` instead
  (diagnostic OR breakage signal, praise-veto scoped to WEAK evidence only), wired calibration-ONLY at
  `source="failure_report"` grade 0.15. **The "repeat request = implicit negative" idea from this same plan
  was also killed**: the 11 near-identical re-asks are dominated by the daily-briefing habit, and labelling
  them negative would fabricate failures. Lesson worth keeping: measure the SUPPLY of a signal before
  designing the detector for it.
- **✅ TIER 3 — DONE 2026-08-01 (later 5) — retroactive negatives from reopened work.** A task REOPENED
  after a turn closed it DONE is a delayed negative on that turn. The missing join was built rather than
  assumed: `tasks.closed_req_id` stamp (consumed on reopen) + `req_id` on CalibrationSample →
  `on_task_reopened` hook → `record_task_reopened_negative` (source `task_reopened`, grade 0.15,
  no-leakage: the closing turn's own components, idempotent, skip-if-already-negative). Task-LEVEL
  DONE→open triggers it — including on ACTIVE projects, which the project_reopened supply measurement
  above never counted, so live supply should run slightly ABOVE the 7/20d estimate. See §6 for the
  review catches (migration-crash guard, honest record() return) and accepted limitations.
- **⏳ TIER 4 — generated labelled probes. HOLD — this exact idea has already failed here once.** Self-play /
  counterfactual / GAIA give checkable outcomes and could be balanced by construction (~50% difficulty).
  BUT the earn-keep harness was CLOSED 2026-07-23 precisely because synthetic batteries ceilinged on this
  35B and could not discriminate (see the §4 header). If revisited: difficulty must be deliberately targeted
  at the 40-60% band, and results kept in a SEPARATE store from production calibration or distribution shift
  corrupts the live fit. Lowest priority; do not start before Tiers 2-3 have measured effects.

**Success criterion (already instrumented):** the fitted model beating `brier_base_rate` on the LIVE corpus —
the `learning_health` line that currently reads "matches the base-rate predictor". Label variance and
per-source counts are reported beside it.

**Two standing hazards, both already bitten once:**
- **Proxy drift.** The Tier-1 graded label scores PROCESS health, not correctness. If the ground-truth tier
  (user corrections) stops flowing, the agent calibrates purely against its own notion of a tidy turn — what
  it already over-indexes on. Keep ≥1 ground-truth source live and separately tracked.
- **Label leakage.** Any new label component must stay DISJOINT from what the feature side uses. Already hit
  twice: `has_tool_error` was the strongest correlate available and useless (it IS a label term), and the
  binary label turned out partly circular with the turn-effort feature — which is why Tier 1 *reduced* the
  apparent edge from 33% to 11% (the 33% was flattering, not lost).

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
  it up at its next restart. **Step 4a (client-facing SSE branch → `_stream_final_generation(self, ss:
  StreamState)`) shipped + LIVE-VALIDATED 2026-07-23:** ~711 lines moved out of handle_chat, verbatim
  (16-space dedent), captures unpacked from a new read-only `StreamState` (26 fields = the exact symtable
  free-var set of the two nested closures). Two transform bugs caught by the suite: the method must be a
  plain `def` (no top-level await — only lazy generators), and `_stream_owns_unregister` is a WRITE-BACK
  (set in handle_chat's frame before the return, else its finally unregisters the streaming turn
  prematurely — the exact class of cross-frame-state bug steps 1–2 warned about; symtable frees only catch
  reads). Suite 9016 passed/0 failed; guards `tests/test_stream_client_extraction.py` (symtable zero-frees
  invariant). Live: prod restart → a `stream:true` request streamed `chat.completion.chunk` deltas → clean
  `[DONE]`, zero errors in ghost-agent.err. **Step 4b (internal stream consumer, ~1,078 lines) — ATTEMPTED
  2026-07-23 and DELIBERATELY STOPPED (not a step-2 clone).** The structural part is easy (uniform 3-way
  control flow: 9 `continue`/7 `break`→returns, fall-through→"proceed"; transform works). What stops it: the
  exact INPUT and REPACK sets need real loop-carried DATAFLOW analysis, and every AST set-heuristic has a
  distinct failure mode — conditionally-bound region-locals (`content`) → UnboundLocalError at construction;
  read-modify-write loop-carried state (`notify_steer_fired`) hidden by symtable's `is_local` → UnboundLocalError
  in-method; and the dangerous one — **cross-iteration repack is SILENT if wrong** (a var written in Region B and
  read on the NEXT turn must be copied back; a naive "read-after" scan misses next-turn reads → the next turn
  sees stale steering/counters, no crash). On the MAIN request path that risk isn't worth the maintainability
  gain, and a green suite wouldn't reliably catch a silent cross-turn bug. **Left Region B inline; decomposition
  stands at 3.5/4 (handle_chat already ~711 lines lighter from 4a).** To revisit: a real liveness/reaching-defs
  pass (live-in=inputs, live-out-across-back-edge=repack), not AST heuristics. See memory `[[agent-py-decomposition]]`.
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

- **[2026-07-22 LLM-stack review — llm.py / router / recording+grammar / delegation / consumers]
  (6 agents). MOSTLY FIXED 2026-07-22 (see §6 "LLM-stack review FIXES" — contention theme, mid-stream
  fail-open, circuit-breaker 4xx, retry classification, router boot landmine, jobs-collect, swarm
  deadline/status, subagent fail-closed, recording, grammar, health, code-gen parity).** Then
  **KV-pin stable-block instability + dream temp-agent containment heal-open FIXED 2026-07-22 (later 2)**
  (§6). Still OPEN (low-value / needs live infra): streaming recording hook + 3 improvements (serializer
  unify, verifier shared prefix, fallback-hint merge). Original catalogue kept below for the record.
  `⊕` = corroborated by ≥2 agents.

  DOMINANT THEME — background/foreground contention on the single main slot (⊕⊕⊕⊕, the biggest
  win if done as one focused pass):
  - **⊕ `targets_main_node` computed from pool *presence*, not routing *outcome*** — `llm.py:1294`
    decides whether to `_wait_for_foreground_clear` from whether a pool is CONFIGURED, but when that
    pool fails a call falls back to main anyway, skipping the foreground-clear → a worker hiccup lands
    unbounded generations on the main slot in direct contention with a live user stream. Same predicate
    also omits `use_coding` (`llm.py:1294-1299`; latent — no live caller passes `use_coding+is_background`).
  - **⊕⊕ consumers omit `off_main_only=True`** so worker-pool failures dogpile main instead of
    degrading: dream/self-play (`dream.py:1559,2300,3213,3479,3618`, `max_tokens` up to 16384, no
    timeout), verifier last-resort fallback (foreground + timeout-less main call, `verifier.py:465-476`),
    `warm_up_workers` (`llm.py:552-557` — a dead worker/critic at boot burns 3× main-slot warmups AND
    can evict the just-warmed main prefix cache; sibling `keepalive_workers` passes it at `:617`),
    project_research/selfhood/workspace critique closures foreground-marked (`project_research.py:331`,
    `main.py:1369,1438`).
  - **⊕⊕ shared `_bg_queue_sem(3)` acquisition is unbounded** (`llm.py:1302`) — critical-path
    `route()` calls (query-expansion, decompose, verify) park behind multi-minute background holders
    (a self-play solver stream holds a permit for its whole generation, `llm.py:1486`); `route()`'s
    12s/45s "fail fast" only bounds the HTTP call, not the semaphore wait.
  - MED — smart-memory extract (`agent.py:4168`) and dream challenge-gen carry NO timeout → ride the
    1200s worker default; a wedged-but-connected Nova pins the journal drain ~20 min/item.
  - MED — circuit breaker counts caller-fault (HTTP 4xx, oversized ctx) and contention ReadTimeouts as
    node failures (`llm.py:964-973`) → 3 contended timeouts open the breaker on a healthy Nova.
  - Suggested fix order: give off-main `route()` its own small sem (or skip `_bg_queue_sem` when
    off-main); add `off_main_only=True` at every dream/verify/critique/warmup consumer; recompute
    `targets_main_node` from the actual chosen route; add per-call timeouts to the untimed background
    calls. One `llm.py` + consumers pass.

  HIGH (data integrity):
  - **⊕ mid-stream stall/break is FAIL-OPEN** (`llm.py:1423-1450` vs `agent.py:11591,12268`) — an
    aborted stream yields a `data: {"error":…}` frame + `[DONE]` and returns normally; both turn-loop
    consumers parse only frames containing `"choices"`, so the error frame is silently dropped, the
    truncated `full_content` is persisted as the final answer and fed to verifier/memory (truncation
    detection checks only `finish_reason=="length"`; an aborted stream is `None`). Fix: consumers check
    `chunk_data.get("error")` and mark the turn truncated.
  - **GHOST_PIN_TOOL_SCHEMAS stable block not byte-stable across turns** (`agent.py:11086-11141`,
    pin ON in prod) — (a) the per-turn skill-playbook query is rebuilt from planner output each
    iteration → a shifted retrieval changes the pinned first user message; (b) the final-gen turn flips
    `tool_header_block` to the slim variant, invalidating the KV cache from message 1 on exactly the
    turn carrying full history → whole-history re-prefill. Verify live via the `Prefill Cache h=` line
    (should be stable within one request; a changing `h=` is the tell).

  ROUTER (built-but-unwired — zero live consumers today, so low operational priority, BUT one boot
  landmine to fix before the planned schema bump):
  - **HIGH — boot-load landmine** (`main.py:1267` inside the broad try whose except nulls the
    dispatcher) — a failed `ComplexityClassifier.load` kills the router AND every retrain path
    (bootstrap at `:1289` never reached); the planned schema bump makes the existing on-disk checkpoint
    raise on every boot → router dead until `checkpoint.json` is manually deleted. Fix: wrap load in
    its own try, fall back to `clf=None` so bootstrap retrains + overwrites. **Do this before any
    schema bump.**
  - The live checkpoint is semantically INVERTED (technical/coding→"easy") from label corruption:
    `n_steps` counts conversation-history assistant messages (`agent.py:14177`), and chess-protocol
    turns flood the "easy" class (`labels.py:84`). Harmless while consumer-less; a schema-bump+retrain
    reproduces the inversion unless labels are fixed first. The 3 logged serve-only feature items
    (`§4B router serve-only`, still unfixed) are dwarfed by this. A/B decision-stamp promised at
    `dispatch.py:46` is never persisted. A hot-swap sanity gate (refuse a model with negative
    jargon/coding weights, `agent.py:3588`) would auto-catch the inversion.

  DELEGATION (cluster FIXED 2026-07-22 for the top 3; residuals OPEN):
  - MED — `jobs(action='collect')` without `job_id` re-returns every retained finished job every time
    (no read-marking; up to 50×8000 chars into a pressure-sensitive context) (`delegate.py:166`).
  - MED — dream's temp agent tool-containment heals open on a dispatch miss (`_rebuild_available_tools`
    re-narrows only via `_subagent_allowed_tools`, which dream never sets; `disabled_tools` checked
    before canonicalisation) → an aliased variant can reach `web_search`/`deep_research` self-play
    disables (`agent.py:2253,6874,7165`). Sub-agent path is safe; dream is the hole.
  - MED — `delegate_to_swarm(await_results=True)` has no overall deadline (`swarm.py:252`, each worker
    ~906s). LOW — failed swarm work lands status `[done]` (`swarm.py:99`); sub-agent tool restriction
    is fail-OPEN (broad try/except, `subagent.py:193`); qwen_bridge cross-loop hazard real but
    UNREACHABLE (`GhostQwenAgent` has zero instantiation sites — best closed by deleting the variant +
    its 3 test files). IMPROVEMENT — two `get_fallback_hint` fns give contradictory hints for one
    failure (`fallback_chains.py:44` + `tool_failure.py:234`); merge into one.

  RECORDING/GRAMMAR (dev features — `GHOST_LLM_RECORD` off, `GHOST_TOOL_GRAMMAR` opt-in — so low
  operational priority):
  - HIGH — the streaming path has NO recording hook and ALL main-model turns stream, so
    `GHOST_LLM_RECORD=1` never captures the calls it exists for (`llm.py:1370-1499`, records only
    non-streamed `chat_completion`/`route`). HIGH — per-process ordinal restarts at 1 while the day
    file appends across restarts; `ReplayLLMClient` stable-sorts by ordinal → interleaves sessions
    (`llm_recording.py:73,98,148`). MED — `route()` fallbacks unrecorded (breaks replay 1:1); grammar
    `sval` prefix-exclusion can't accept a value ending in a `</parameter>` prefix → runaway
    generation; name+param join collision `p-web-search-query` (`tool_grammar.py:63,78,148`). LOW —
    replay loader dies on a torn last line; GHOST_HOME frozen at first record. IMPROVEMENT — recording
    has no size guard (base64 vision payloads balloon the JSONL to GB); feed grammar output to a real
    GBNF parser in tests.

  LLM REQUEST-SIDE (lower severity):
  - MED — verifier auto-repair on a project-wrap-up builds a self-contradictory payload (full XML tool
    schema + "RUN it now" while native tools suppressed and the turn forced-final → repair can't comply,
    burns a `_MAX_VERIFIER_REPAIRS` slot) (`agent.py:11073-11197,13166`). MED — `_execute_post_mortem`
    swallows the transient-requeue exception → a post_mortem item on a worker timeout is popped and
    permanently dropped (the 2026-07-09 requeue fix covered only `run_smart_memory_task`)
    (`agent.py:4357`). MED — `_aa_code_gen` idle twin still `max_tokens=1024` where
    `default_code_generator` was raised to 4096 for the inline-`-c` truncation bug (`agent.py:3815`).
    IMPROVEMENTS — retry classification inverted (`ConnectTimeout` not retried though it's the only
    guaranteed-nothing-sent failure, `llm.py:1216`); `warm_up_workers` fires the 3 warmups sequentially
    (same slot reused, other `-np` slots stay cold — `asyncio.gather` them, `llm.py:543`); unify the 5
    hand-rolled node-payload serializers + kill the `errors='ignore'` footgun (`llm.py:896,955`);
    surface `NodeCircuitBreaker.get_status()` in `/api/health .nodes`; share a common leading block
    between verifier stage-1/stage-2 prompts so llama.cpp prefix-cache hits (`verifier.py:202`).

- **[2026-07-20 three-stack review — project-autonomy + turn-loop + code-correction] ALL FIXED
  2026-07-20 (later 3)** — see §6's "three-stack review FIXES" entry. 10 parallel fix agents (strict
  one-file-per-owner) + coordinator on agent.py/main.py/slack; full suite green. Findings below kept
  for the record; `⊕` = independently corroborated by ≥2 agents. The `TurnState`/`FinalizeState`
  decomposition seam audited CLEAR (no `locals()` recurrence; MUTATED_FIELDS repack symmetry confirmed
  by two agents). **Headline: the streamed-finalize bypass (H1) had invalidated the Round-10
  streamed-trajectory fix — now correctly closed by mirroring the work_log + trajectory writes into
  the stream drain via a shared `_write_project_work_log_safe` helper.**

  CRITICAL:
  - **[fs/destructive] `manage_projects action=cleanup project_id="../.."` escapes the sandbox** —
    `cleanup` is the one destructive action skipping `_resolve_project_ref`; `_canon_id` only
    strip/lowercases, so `../..` survives into `workspace_cleanup._project_dir` →
    `Path(root)/"projects"/"../.."` resolves to the sandbox PARENT (reproduced), and
    `tidy_project_workspace(min_age_hours=0)` `os.walk`s it deleting dotfiles/`.git`/logs with no
    containment re-check. Fix: validate the id against the store (or reject `..`/separators) before
    tidy, and add a `root.is_relative_to(<sandbox>/projects)` assertion in tidy/sweep.
    `tools/projects.py:2071`, `core/workspace_cleanup.py:60-100,325`.
  - **[fs/correction] replace→write auto-promote overwrites a file with an inner snippet, SUCCESS** —
    `extract_code_from_markdown(str(old_text))` called WITHOUT `filename=`, skipping the
    "whole input already parses → keep it whole" guard; a module with a fenced docstring example is
    replaced by just the snippet, syntax-check passes, status SUCCESS. `tool_write_file:2160` passes
    `filename=` correctly. Fix: forward `filename=filename`. `tools/file_system.py:955`.

  HIGH:
  - **[project/streaming] ⊕⊕⊕ streamed forced-final turns bypass the entire finalize tail** — they
    `return` the SSE generator at `agent.py:12019` before `_finalize_and_return` (13164), so
    `add_work_log` (only caller finalize:9454) and `_record_turn_trajectory` (only caller
    finalize:9480) NEVER run. Web UI always streams AND one-task-per-turn `force_final_response`
    guarantees this ending on any task-closing turn → no work_log (re-opens the 07-18 "forgets
    project work" gap) and NO trajectory recorded. **The Round-10 streamed-trajectory backfill is
    inert** (parking setter 14137 gated on non-str `final_content` its only caller never passes;
    backfills a trajectory never created). Fix: mirror work_log + trajectory-record + correction-stash
    into the stream drain (11324-12010), not the backfill; add a streamed-turn integration test
    asserting both are written. `agent.py:9404/12019/14136`.
  - **[project/store] ⊕⊕ `update_project(metadata=…)` is a blind whole-dict replace** — wipes
    `design_ledger`/`config`/`steps_used/cap`/`defect_reopens`/research index/runtime counters. The
    documented budget-raise (`action=update metadata={"steps_cap":100}`, steered by the batch blurb)
    destroys project state; the advancer's non-atomic RMW also clobbers cross-process
    (`_atomic_metadata_update`) writers. Fix: route metadata mutations through a merge/atomic path.
    `memory/projects.py:329`, `tools/projects.py:1490`.
  - **[project/advancer] ⊕⊕ IN_PROGRESS leaf orphaned on deploy-by-kill** — claim committed before
    ~6 awaits, `next_ready_leaf` eligibility `{PENDING,READY}` only, handlers catch `Exception` not
    `CancelledError`, no boot reaper. Plain SIGTERM mid-tick wedges the task forever (project stuck
    ACTIVE, "all complete"). Fix: boot-time reaper resetting stale IN_PROGRESS→READY, or add
    IN_PROGRESS to eligibility with an age guard. `project_advancer.py:597`, `planning.py:789`.
  - **[turn-loop/context] `_prune_context` third un-capped return** — the `len(non_system_msgs)<=5`
    branch returns the tail verbatim (no `_cap_oversized_tail`); a few huge tool results still ship
    oversize → xrick HTTP-400 → destructive recovery. The 07-18 fix wrapped only 2 of 3 returns.
    `agent.py:2518-2533`.
  - **[verify/fail-open] grep-exit-1 → `EXIT CODE: 0` defeats autoadvance verify** —
    `execute.py:654` rewrites grep-family no-match to exit 0; advancer `_looks_like_failure` reads a
    pass, so `grep -q "marker" file` marks DONE when the marker is ABSENT (theatrical completion).
    Correct for the interactive strike loop, fail-open for verify — the verify path needs the raw
    exit code. `execute.py:624`, `project_advancer.py:1190`.
  - **[parser/native] leaked-framing repair truncates legitimate content, SUCCESS** —
    `_value_has_leaked_framing` fires on any `file_system` write whose `content` holds a two-param
    tool-call example (`</parameter>`…`<parameter …>` — produced when writing docs/tests about the
    tool format); repair truncates content at the first `</parameter>` and writes the fragment.
    Native-ON default; content in `arguments` is never tag-checked. Fix: length/known-tool guard, or
    skip repair when the primary tool name is valid and args parse. `agent.py:1553`.
  - **[fs/correction] streaming-replace (>1 MB) bypasses the guard chain** — no marker-leak check,
    no syntax rollback; merged-args markers (`<<<< SEARCH`/`====`) written verbatim, SUCCESS. Fix:
    route the streaming path through `_write_replace_guarded`. `file_system.py:1031-1071`.
  - **[cleanup/destructive] recurring tidy deletes `.git/`+dot-configs on ACTIVE projects, no age
    gate** — the `TIDY_MIN_AGE_HOURS` gate applies only to the media branch; `_is_debris('repo/.git/
    config')`/`.eslintrc.json`→True unconditionally. Idle watchdog corrupts in-flight clones. Fix:
    apply the age gate (and a registered/referenced check) to debris too. `workspace_cleanup.py:114`.
  - **[cleanup/destructive] registered absolute-path deliverable unprotected → DONE sweep deletes
    it** — `deliverables=["/workspace/projects/<id>/x.png"]` normalizes to `workspace/projects/<id>/
    x.png`, matching no walked rel. Fix: handle the `workspace/` prefix in `_normalize_rel` +
    `register_file_artifact`. `workspace_cleanup.py:77`, `memory/projects.register_file_artifact`.
  - **[project/api] `POST /api/projects/{pid}/advance` = theatrical completion** — no tool_runner
    passed; research leaf closes DONE `result_summary="(no tool runner)"`, budget charged, events
    stamped to the wrong project. Fix: build the default runner (as the docstring promises) or refuse
    non-idle advance. `api/projects_routes.py:179`, `project_advancer.py:821`.

  MEDIUM (project autonomy): `task_update` with `result` but no `status` is a success-shaped no-op
  (`count:1`, writes nothing) and the gate-recovery text tells the model to make exactly that call →
  gate-clear loop (`projects.py:1794`); bulk `task_update` constraint audit blocks fileless SIBLING
  tasks (`1767`); `reconcile_conversation` fails open on a missing/LRU-evicted sentinel (50-entry/24h)
  → stale project active cross-conversation (`444-527`); API `switch`/`resume`/`delete` set
  `current_project_id` directly, bypassing `_set_current` → sentinel desync / reactivates a deleted
  id (`projects_routes.py:142-167`); ⊕ dead ReplanBridge (revises a throwaway tree, nothing persists,
  counts "revised") (`planning.py:475`, `metacog.py:141`); ⊕ idle round-robin starvation —
  blocked/idle ticks never re-stamp `last_autoadvance_ts` (`project_advancer.py:356`, `agent.py:3703`);
  HUMAN_GATE not enforced on the interactive close → FAILED not NEEDS_USER (`planning.py:115`,
  `project_safety.py:104`); now-live plan gate per-char garbage on string `postconditions` + crash on
  `status:null` (`planning.py:146`); `_maybe_rollup_project_status` non-transactional, no `AND
  status=?` guard → can stomp a concurrent ARCHIVE→DONE + fire cleanup (`memory/projects.py:677`);
  reactivation dead-end for FAILED/PAUSED/NEEDS_USER (no schema-legal ACTIVE) (`memory/projects.py:548`);
  `delete_task` never rolls up (`725`); metadata array-string poisons `dict(metadata)` readers
  (`projects.py:1087`); `parent_id` + `resume`/`get`/`switch` title args not canonicalized/honored
  (`1436/1558/1601`); `record_runtime` only on success → runtime rail blind to failed builds
  (`project_advancer.py:414`); digest scans only ACTIVE → hides the same-tick NEEDS_USER/DONE rollup
  (`project_digest.py:51`); sync `shutil.rmtree`/sweep on the event loop in async routes
  (`projects_routes.py:136`).

  MEDIUM (turn-loop): post-batch failure bookkeeping attributes to the LAST tool in the batch, not
  the failing one (strike sig / fallback hint / failure-dimension all mis-keyed → ≥3-repeat breaker
  fires late/never) (`agent.py:8054`); `is_mutating` drifted from `is_sandbox_mutation` —
  `unzip`/`git_clone` non-mutating so world-changed reset never fires post-clone + batch-dedup can
  collapse them (`6965`); batch dedup collapses identical `browser`/`manage_projects` calls (two
  clicks execute once) (`7278`); per-turn injection (~10-20k tok: schemas+hydration+briefing)
  uncounted at the prune/steer threshold → overflow band with no steer (`10799`); first-read budget
  exemption bypasses the occupancy-shrunk cap by one whole-file read (`file_system.py:814`); raw
  uncapped scratchpad injected every turn, invisible to every governor estimate (`agent.py:11022`);
  `_apply_file` writes from a truncated/absent snapshot → clobber, and same-path double-append
  discards the first (`coding_executor.py:513`); retry re-renders a stale pre-edit excerpt → burns
  all 4 attempts, half-applied file on disk (`coding_executor.py:640`); `_format_error` hardcodes
  `EXIT CODE: 1` (erases timeout 124) (`execute.py:179`); egress-guard / spill-log return
  success-shaped prose the verify classifier passes / point at an unreadable path
  (`execute.py:158`, `docker.py:724`).

  MEDIUM (code-correction): REPLACE closer not line-anchored (a `>>>>` in the payload truncates it)
  (`file_system.py:1110`); git-conflict content in SEARCH text hijacks the `=======` separator
  (`1109`); lone-surrogate content truncates the target to 0 bytes + the ValueError returns with no
  `Error:` prefix so `coding_executor` records it written (`2172`); indented markers evade the leak
  guard (`1981`); `projects/<slug>/` prefix strip collapses an explicit FOREIGN-project path into the
  active one (`348`); `tool_rename_file` silently clobbers an existing destination (`2729`);
  `_apply_edits` after/before-insert uses 2-arg `.replace` (all occurrences) (`coding_executor.py:466`).

  LOW (sample; full list in agent transcripts): promotion nudge counts reads as "sandbox writes"
  (`agent.py:9348`); work-log paths lower-cased → ENOENT on case-sensitive FS (`7682`); `_SELF_REF_RE`
  misroutes "how do I…" research to introspection (`project_advancer.py:454`); binary media crowds
  the 12-file executor snapshot (`67`); substring library-match mints false cross-project edges
  (`project_concepts.py:129`); `detect_contradiction` flags agreeing summaries (`project_safety.py:121`);
  dangling alternative id consumed + blocks parent with valid alternatives queued (`planning.py:280`);
  id-less native tool-call dup collision (KNOWN §4B, `id=""` default confirms reachable)
  (`agent.py:12240/13994`); `unescape_xml_values` double-decodes native args (`6731`);
  truncated-tool-call partial body dispatched w/o recovery steer (`6076`); `GHOST_TOOL_GRAMMAR=1`
  without `--no-native-tools` attaches contradictory constraints (`11248`); `rg` pattern passed
  positionally (a `-`-leading pattern parses as a flag) (`file_system.py:2552`); root-cwd retry
  adopts result unconditionally, replacing a real traceback (`execute.py:589`); heredoc-into-interp
  egress-guard bypass (`execute.py:44`).

  FIX-ORDER RECOMMENDATION: (1) H1 streamed-finalize bypass — biggest blast radius, corrects the
  inert Round-10 fix; (2) C1 + H8/H9 cleanup containment/age-gate/deliverable-normalization (one
  focused `workspace_cleanup.py` pass); (3) C2/H6/H7 file_system write-corruption trio; (4) H5+H10
  verify fail-open + H3 IN_PROGRESS reaper; (5) H2 metadata whole-dict replace; (6) H4 prune return,
  then the medium tier.

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
- **sandbox/docker.py + services.py + execute.py** — client-side exec deadline, infra-error marking,
  resume-stopped-container, adopted-published-ports, spill-counter, heal-retry safety, port-reclaim
  ownership + container-generation stamp + atomic restart **ALL FIXED 2026-07-22 (later 4)** (§6). Still
  OPEN: remove-while-exec race (self-heals, low; do NOT auto-retry — not idempotent); readiness
  false-negative removes a healthy container (self-heals); the full infra exit-code sentinel (light
  marker done). **Linux-only** (prod is macOS/bridge): exec-user vs root-provisioned env
  (Playwright/Chromium in /root mode 700, sudo refuses unknown uids — needs `useradd -o -u <host_uid>`
  + browsers to a world-readable path + Dockerfile marker bump); `HOST=0.0.0.0` host-mode LAN exposure
  (`binds_host_netns()` accessor added — services-side `HOST=127.0.0.1` export pending); egress-guard
  loopback bypass (`[::1]`/decimal IPs; host-netns only).
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
- **interface external GPU servers + clients** (node-deployed; SKIPPED per operator 2026-07-22 — "not
  in use"; reviewed 2026-07-22 later 4, §6). The **image server (Jetson) is already hardened** (07-15:
  constant-time fleet-key auth, GPU thread, VRAM budget) — only a prompt-token cap is missing. The
  **voice server (Orin) is the open CRIT**: NO auth on `/stt`+`/tts` bound 0.0.0.0, sync GPU inference
  on the async loop, unbounded upload into RAM on an 8GB box, no `/health` — fix is to port the image
  server's `_require_key` + GPU-thread + input-cap pattern (~35 lines) + client `X-Ghost-Key`. uConsole
  client: SSE `aiter_text` SILENTLY LOSES tokens (persisted into history), camera+60fps QTimer leak on
  Escape, send-race interleaves streams, unbounded base64-image history. Slack bot: owner-lock HOLDS,
  but thread-history admits any third-party bot (`or msg.get("bot_id")`) into context + owner
  `file_share` DMs dropped as unauthorized. (Voice/client fixes are per-device deploys.)
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

### 2026-08-05 — three token slots, zero producers: upstream `usage` captured; tool-choice fidelity re-measured (0.821, and the CLUSTERING CLAIM DIED)

**1. Tool-choice fidelity re-measured — the 0.772 was two things wrong, and the clustering was a
third.** Triggered by the operator's instinct that the number was "poisoned by a bug that is now
fixed". It was not that bug — but it was wrong.
- *The parser-leak corruption never reached it.* Verified two ways: the miner's era cutoff
  (`2026-07-31T19:15`) postdates the last occurrence (~18:54, all 17 dated pre-fix on 08-04), and
  an independent scan of all **533 tool-call arguments** in the freshly mined pool with
  `utils/leaked_framing` returns **0 hits**. The report's own `corpus purity` header agrees. What
  the corruption DID pollute was the macro/n-gram counts, closed 08-04.
- *The real defect in the citation was metric choice.* `0.772 = fidelity_runner` (44/57) counts 5
  rows that failed to REPLAY as wrong — deliberate, so a GEPA candidate cannot win by skipping hard
  fixtures. `fidelity` (replayable-only) was 0.846. `tool_ontology.py:198` says outright *"neither
  is 'the' fidelity"*; quoting the optimizer's gaming-resistant training metric as a measurement of
  the toolbox was the error.
- *Fresh measurement:* **0.821 (55/67), 0 unreplayable** — so both metrics agree for the first
  time and there is no metric to argue about. Mined 2026-08-05, 197 fixtures / 71 positives.
- ⚠ **THE CLUSTERING CLAIM DOES NOT REPRODUCE.** 12 misses land on **12 distinct pairs, max count
  per pair = 1** (two are no-tool stalls). The §4F premise that the misses "cluster into a handful
  of specific pairs (browser↔file_system, execute→file_system, manage_projects over-selected)" —
  which is the whole evidential basis for "the boundary is drawn in the wrong place" — is not
  visible on 67 held-out cases. Honest caveat: 12 misses over 16 tools is ALSO too few to rule a
  boundary problem out; perfect scatter is what you would see either way. **The ontology question
  is open on evidence, not leaning either way.** `docs/self-improvement.html` updated.
- *Reproduce:* `mine_tool_fixtures.py --private-pct 95 --force-write` then
  `optimize_tool_descriptions.py --smoke --force-supply --confusion-out …` then
  `tool_ontology_report.py --replays …`. **Gotcha:** `--private-pct 100` fails with
  `degenerate public/private fixture split` — the runner requires both tiers non-empty even under
  `--smoke`, which only scores private. 95 gives 67 private / 4 public. The default 30% yields
  **14** private positives (7.1 points per case; the miner prints `TOO COARSE`). Widening is
  legitimate here because `tool_selection.pick` has **never been promoted** — no artifact on disk,
  so the incumbent has never seen any of these fixtures.

**2. Upstream token accounting — a field that read 0 on 1515 of 1515 records.** Found while
looking for non-GEPA work. THREE token slots existed and NONE had a producer:
`Trajectory.tokens_in/out` (0 of 1515), `eval.TaskResult.tokens_used` (only producer:
a hardcoded `0` in `scripts/eval_baseline.py:91`), and the upstream `usage` block, which nothing
read anywhere in `src/`. Every `SuiteResult` ever written therefore reported
`total_tokens: 0` — summed, serialised, and sitting beside real numbers. **The paired ablation
could only say "~1.8× latency" because the token half was never measurable.** Same shape as the
mock-guard and the write-only optimizer: computed, aggregated, reported, structurally always zero.
- *Captured at all three funnels.* `chat_completion` (both branches), `route` — **missed on the
  first pass and caught by the fresh-eye review**; it serves verify / decompose / classification,
  so omitting it under-reported the turn silently and in the direction that flatters it — and the
  streamed path, which needs `stream_options.include_usage` (an OpenAI-compatible server sends no
  usage on a stream without it) plus a parse of the final chunk, whose `choices` is empty.
- *Per-REQUEST sum, not per-call*, in a bounded 32-entry LRU on the client. `usage_for()` returns
  `{}` for an unknown request rather than zeros — "never measured" and "a cheap turn" must stay
  distinguishable, that being the whole defect.
- *The streaming hook is deliberately OUTSIDE the `if _rec_on:` gate* — `_stream_rec_accumulate`
  only runs under `GHOST_LLM_RECORD=1` (off by default), so folding usage into it would have
  shipped it dark. Pinned by an AST test that also fails if the gate is renamed, so it cannot pass
  vacuously.
- *Surfaced on `/api/chat`* as an OpenAI-shaped `usage` block (+ `cached_tokens`, `ghost_llm_calls`),
  omitted entirely when unknown. Needed because the eval runner reads Ghost's API, not the
  upstream — without it the `eval_baseline` fix would have been a second silent zero.
- *`prompt_tokens_details.cached_tokens` carried through.* The prefill cache is reported in the log
  as a CHARACTER count (`chars=16757`); this is the first real hit measurement.
- *Two traps, both caught by the suite (11 failures across 4 modules):* a `MagicMock` client
  returns a **truthy** mock whose `.get()` is also a mock → serialised to a 500 instead of a reply
  (fix: `isinstance(dict)` + `int()`, never truthiness); and stream chunks are `str` on
  `aiter_lines()` but `bytes` elsewhere → `TypeError` in the stream loop.
- Tests: `tests/test_llm_token_usage.py` (25). Suite **11233 passed**. Docs: `core/llm.html`,
  `api/routes.html`, `troubleshooting.html` (reading `cached_tokens` to diagnose slow turns).

**3. `--metacog-mem-high` default 85 → 97.** Ghost always runs co-resident with a local LLM server,
which pins RAM as steady state; 85 treated that resting condition as pressure. Real pressure is
caught by the free-RAM conjunct and the hard floor, which fire on absolute numbers. **Live data
that prompted it:** RAM ranged 52.4–98.0% with free 744 MB–17.5 GB, and produced **1** signal
(info) against **15 CPU** signals (14 info, 1 warning) at the CPU default of 85 — i.e. the noisy
metric is CPU, not RAM. The operator's launcher already passes 98, so this changes nothing live;
it fixes the out-of-box default. Also removed a hardcoded `85.0` fallback in `main.py`'s signal
renderer — the block whose own comment exists *because* of a stale-threshold reporting bug.
⚠ *Open, not acted on:* `--metacog-cpu-high` is still 85 and is what actually fires. Operator call.
⚠ *Noted, inert:* `HostSnapshot.healthy` compares against CLASS defaults, ignoring configured
thresholds (documented limitation) — harmless today because it has **zero consumers**.

### 2026-08-04 (later) — THE FIX BROKE IT: arming four never-run subsystems destroyed 13 lessons and opened two live redaction leaks

**Origin.** The audit below ended with a `component_guard` fix that ARMED four subsystems which
had never executed in production. Two follow-up fresh-eye reviews were run on those changes
before the operator went idle. Both came back with CRITICALs *in the fix code itself*.

**What the arming actually did.** `prune_low_utility` — destructive, unattended, vector-twin
deleting — ran for the first time in its life at 07:10 and again at 07:36, dropping **8 then 5
lessons** (playbook 50 → 38) and logging only a count. Its archive-before-delete safety net had
been written at 07:29, i.e. *after* the 07:25 process start, so the live process never loaded it.
One victim scored `retrievals=277 succ=77 fail=32` — a 70%-success lesson pruned as "low utility".

**Why it will stay off.** The cutoff is a RELATIVE bottom quartile, so it always finds victims
however good the playbook is; and failure-distillation mints lessons at utility ≈0.77 against a
measured live cutoff of 1.1183, so every distilled lesson is structurally guaranteed to be deleted
once it reaches `min_retrievals` — two subsystems fighting each other, both "working as designed".
`GHOST_SKILL_PRUNE` now defaults **off**; the archive **fails closed** (an unwritable archive
aborts the prune rather than warning and deleting anyway — the first version lost 7 more lessons
in a probe); quarantined rows are exempt (they decay into the bottom quartile *by construction*,
since quarantine stops their retrievals accruing).

**Partial recovery.** 5 of the 13 were reconstructed from `GHOST_LLM_RECORD` prompts written
before 07:10 — rendered lessons carry TRIGGER/ANTI-PATTERN/CORRECT-PATTERN — and written to
`skills_pruned_archive.jsonl` rather than re-injected, since whether they deserve reinstatement is
a content judgement. The other 8 are unrecoverable; the only backup is 2026-07-16, and its 36-row
delta spans three weeks of legitimate churn, so the victims cannot be attributed out of it.

**Two live redaction leaks, both introduced by a false-positive fix.** The `ipv4` rule's new
lookbehind excluded `/` and `-`, which spared **every URL host, every path-embedded address, and
the second address of an `a-b` range** — including the only genuine host on this box, which
appears as the Flask `* Running on http://<LAN-IP>:5055` line the narrowing was measured against. The `credit_card`
dot-guard vetoed any trailing dot, so a card at the end of a sentence escaped **entirely**. Both
fixed by moving the judgement out of the regex and into `_ipv4_repl` (URL/path vs the `Name/1.2.3.4`
product-token shape) and by narrowing the card veto to dot+DIGIT. The cue-word list also lost
`rule`/`item`/`step`/`table`/`v`, each of which doubles as a config key: **when a redactor's two
error directions conflict, the leak is the one that matters.** 24-case probe, 0 failures.

**Self-play was one pick away from `DROP TABLE` on the live database.** The journal stash held 20
un-replayed records; one was a real past user turn reading *"Run this EXACT SQL via postgres_admin
… SELECT 1; DROP TABLE web_order_line_options_old;"*. Self-play replays mined user messages
VERBATIM to a solver holding the real toolset, at `journal_prob=0.75` once the frontier saturates
(6 of 8 clusters currently are). Only the generic multi-statement validator stood in front of it.
`_is_unsafe_challenge` now refuses destructive DDL/DML/shell shapes and the "run this exactly, do
not modify it" framing at synthesis time — blocking 1 of the 20 live records, the right one.

**Also.** The manifest backfill was writing DB manifests into RELEASED (chmod 0555, immutable)
projects while its paired `PROJECT_MAP.md` write failed at `logger.debug` — DB and disk
permanently disagreeing while the INFO log reported success; it now excludes RELEASED, reads a
bounded 4 KB instead of whole files (a registered 483 KB bundle cost 483 KB to describe 1.5 KB of),
and skips binaries rather than storing a hallucinated description of `SQLite format 3\x00…`.
The episode success label moved from `_fails == 0` to `_fails < 6`: `execution_failure_count` is a
**decaying strike ledger** (a System-3 pivot subtracts 2, each clean success subtracts 1), so
`== 0` meant "no strikes outstanding", which a turn that failed six times and then recovered also
satisfies — wrong in both directions.

**And the numbers in my own write-up were wrong.** The claimed verify_bench improvement
(7 cases/49 trials/0.0455 → 21/155/0.0132) reproduced on none of its six figures; the real
measurement at the default `--private-pct 30` is **4/30/0.0833 → 29/220/0.0093**. Corrected in
§4F and `docs/self_improvement.md`, with the bad figures named rather than quietly replaced.

**Then the fixes to the fixes were reviewed, and two more rounds fell out.** Same instruction —
executable probes, re-derive every number.

*Round 2 (optimizer/bench).* `--refresh-mined` would have **destroyed the pool on a zero-yield
mint and exited 0** — and silent extraction failure is precisely this pipeline's characteristic
bug (a retuned template once matched 0 of 580 records with an identical opening sentence). It now
refuses an empty mint or a >50% shrink. The aggregate ship-gate still did not model the read-site:
that sums over the per-request `_intent_filter`ed subset, not a fixed list, so a set totalling
19,998 (pass) reaches 20,320 and applies **zero** once `postgres_admin` is filtered out — which is
the default config. The gate is now **worst-case over subsets** (per-tool deltas clamped at 0,
runtime-only artifacts charged in full). `bench_provenance`'s new `judge` field was **always
empty** — read off the `Verifier`, which holds no endpoint — i.e. verification theatre in the block
whose entire job is deciding comparability; it now resolves through the client and says
`unresolved` rather than asserting a blank model. And `optimize_verifier` trains on the **public**
tier of the same mined pool `verify_bench` loads by default, so a post-optimization bench measured
partly on cases the optimizer saw: `--tier private` added, overlap printed.

*Round 3 (safety/redaction).* The IPv4 product-token exception was **structural** ("a product
token is a word not preceded by a separator") and leaked broadly — any path segment containing
`-`, `_`, `.` or `:` defeated it, so `/var/log/my-app/<ip>`, `/opt/ghost_agent/<ip>` and
`http://example.com/<ip>` all passed through verbatim; 36 of 83 real directories on this box were
leak-triggering prefixes, and a 39+ char segment leaked regardless of content. There is no
structural difference between `logs/10.0.0.9` and `Chrome/120.0.0.0` — **only the name** — so the
exception is now an explicit UA allowlist. The cue scan also crossed newlines, so a markdown
heading suppressed redaction on the next line. And the leading `\b` required a non-word char before
the first octet, which a JSON-escaped `\n` does not provide: seven addresses sat in the clear
inside serialized tool output, which is exactly where addresses live. **Live sweep after: 1130 of
1130 non-loopback quads redacted across 144 MB, zero false positives.**
Two more: my widened quarantine block set added `(trigger, "")` for a row with one content field
empty, silently restoring **trigger-only blocking** — the un-learnable-topic bug it was fixing; and
the archive-before-delete invariant covered only the prune, while `retract_lessons_from_trajectory`
(four live call sites) deleted with no record at all. Both now fail closed through one shared
helper. The credit-card rule was corrupting 95 of 147 Luhn-valid live runs — none of them cards —
by matching inside hyphen/pipe-delimited identifiers (Unsplash photo ids, epoch-millis, LinkedIn
activity ids); boundaries now exclude `-`, `_`, `|`.

*And the episode label was still wrong.* `_fails < 6` missed the cap's **other arm**
(`exec >= 6 OR total >= 8`), so a turn capped purely on transient failures emitted the "I hit a
hard limit" sentinel and was stored as SUCCESS — the exact defect the rewrite claimed to kill,
surviving through the arm nobody checked. `transient_failure_count` is not in scope at either
finalize site, so the cap now records itself through `core/turn_facts`. The threshold moved 6 → 3
to match `is_complete_failure` in the post-mortem gate: at 3-5 the system had been saying COMPLETE
FAILURE (post-mortem), saturated (`risk.py`), negative (`confidence.py`) **and** success (episode
store) about the same ledger value.

**Three reviewer-caught vacuous tests**, all mine: a two-address assertion satisfied by the first
address while the second leaked verbatim; a quarantine-prune test that quarantined rows the prune
was never going to pick; and a `correct_pattern` guard no test could distinguish. Each now
revert-verified red. Reverting the ipv4 fix turns **15** tests red where it turned 4 red before.

**Method note.** Every defect in this entry was in code written the same day *to fix a defect*,
and every one was past a green suite — across three successive rounds. The reviewers were told to
prefer executable probes over reasoning and to re-derive every load-bearing number in the new
comments, which is how the non-reproducing figures surfaced; roughly a dozen of my own stated
numbers were wrong and are now either corrected or explicitly retracted in place.
**Suite: 10858 passed / 14 skipped.**

### 2026-08-04 (later 2) — LIVE FUNCTIONAL TEST: escalation overturned a CORRECT refute and laundered a fabrication into `passed`

**Origin.** Operator, after the restart: *"run functional tests against the agent at port :8000 to make
sure everything works as it should after all these changes."*

**Everything built this session works.** Health ok (worker circuit `closed`, 0 failures). A basic turn
passed in 49 s with exact instruction compliance. A tool-using turn exercised the whole new path and
produced the first live proofs: `GEPA: loaded tuned instruction for 'verifier.adjudicate' (5366 chars)`
(the read-site firing in production — the activation evidence that counter exists for), the first
`record_escalation` row on disk, and learning-health rendering it as
`claim/refute: 1 escalations — 100% overturned (1/1 decided)`. Also confirmed live: the
`tag failure-dimension` label fix (7 lines since boot, replacing the "classify failure" text that read
as an error).

**And it caught something the whole §4F reading of the overturn rate depends on.** Request
`03b96c28` (trajectory `f78c8b33`): the probe asked for a line count on a path OUTSIDE the sandbox.
All THREE tool calls failed correctly (`file_system`, `execute`, `file_system` — "does not exist in
the current project's sandbox"), strikes counted 1/6 and 2/6, the turn graded **failed**. The agent
then answered **`0`** — a 1-character fabrication with no acknowledgement of failure. The cheap judge
**REFUTED it, correctly**, at confidence 1.0. `_escalate_refute` sent it to the main model, which
**OVERTURNED to CONFIRMED** (0.8), and the outcome was rewritten:
```
turn outcome — CORRECTED failed → verified (late verdict)
verifier — late verdict backfilled into the corpus: trajectory f78c8b33 → passed
```

**Why this matters beyond one turn.** The 84% overturn rate has been read throughout §4F as "the
cheap judge false-alarms and the strong model corrects it" — that reading is why refute escalation
is on and its churn is treated as latency-only. This is the opposite case: the cheap judge was
RIGHT, the strong model was WRONG, and the error propagated into the learning corpus with the
outcome label flipped. One case is not a rate, but it is an existence proof that some share of the
~8/day live overturns may be corrupting labels rather than repairing them.

**Not a bug in the new confirm guard.** `_escalate_confirm` exists precisely to stop a high-stakes
CONFIRMED from laundering a structural failure into a pass, and it deliberately skips verdicts
carrying `escalated_overturn` — re-escalating would just re-ask the same judge. So the
refute→overturn path reaches the exact outcome the guard was built to prevent, through a door the
guard correctly declines to close. **The structural gap: once the main model says CONFIRMED, nothing
checks it.**

**Instrument shipped: `scripts/escalation_audit.py`** (read-only). Joins the ledger to trajectories
via `iter_trajectories()` (corrections overlay applied — the question is what the corpus ENDED UP
believing) and prints an adjudication card per overturn, flagging `tools_failed` as the population
where a fabrication becomes a pass. `--json` feeds a judge pass. **Run it after the ledger fills:
if overturns on tool-failed turns are a material fraction, that outranks the optimizer work**,
because the corpus has been absorbing false passes for as long as escalation has been on.

**Also surfaced:** the audited trajectory reads outcome `passed` WITH `failure_reason: "structural
failure"` still stamped, although the upgrade path at `agent.py:7124` clears the reason — so either
a second path performs the upgrade or the corrections sidecar re-applies the outcome without
clearing the reason. `passed` + `structural failure` should be an impossible pair.

**Suite hygiene, measured:** `GHOST_API_KEY` does NOT change the suite (byte-identical 11046/14 with
and without — the old "set it for full-suite runs" note no longer reproduces). The 14 skips are
deliberate: 13 are tests for removed/disabled features (4 loop-breaker, 3 System-2 planner,
2 rambling-guardrail, 4 singles) and 1 is a real gap (`llama-cpp-python` absent). 14 is the healthy
steady state; a different number is the news.

**→ OPERATOR DECISION, same session: "add the shape rule, structural failure shouldn't pass"** —
narrowing the 2026-07-31 honest-failure rule. ✅ **SHIPPED, see below.**

**═══ THE SHAPE RULE (2026-08-04, operator decision) ═══**

**Rule:** if a turn's tool calls **ALL** failed AND the final response does not acknowledge the
failure, a `passed` verifier verdict must NOT upgrade it out of FAILED — **whichever model produced
that verdict.** Shape-only: it asks no model anything, which is what keeps it outside the judgment
loop that produced the defect.

Canonical: `distill/outcome_heuristics.py::unacknowledged_total_failure` (+ a
`_for_trajectory` adapter), consumed by `resolve_turn_outcome` as **rule 2b**. Kill switch
`GHOST_UNACKED_FAILURE_GATE`, default 1, read in ONE place that every site routes through — so it
cannot be live on one path and dark on another (there is a test asserting the switch is read once).

**Two properties that keep it from becoming the opposite mistake:**
- **ALL, never ANY.** A turn where one tool fails and the agent recovers through another is a GOOD
  turn and keeps its PASS.
- **Non-manufacturing.** Rule 2b can only WITHHOLD a pass, never invent a FAILED, and fires only on
  turns already structurally failed. This deliberately bounds the corpus sniffer's false positives.

**The acknowledgment detector was MEASURED, not asserted:** hand-labelled against all 23
all-tools-failed turns in the live corpus — **15 acknowledged-correct, 8 silent-correct, 0 false
acknowledgments, 0 missed**. Judged on CONTENT not length (`"not found"` acknowledges; a
five-paragraph chess reply does not). Calibration removed bare `blocked`/`empty`/`sandbox` after a
chess reply ("your bishop gets blocked by e3") false-acknowledged. **The hard case was `NOPE`** —
the 07-31 rule's own live-validation probe replies with a user-pinned literal, and a pure
content detector would call that a fabrication, regressing the exact case the operator validated.
Narrow shape-only escape: an explicit `reply/respond/say … exactly|only|just …` span in the REQUEST
licenses a ≤64-char whole-token echo. `"reply with just the number"` names a FORMAT, not a literal,
so it does **not** license `0` — that discrimination is a test.

**Verified independently (not taken on report):** the real `f78c8b33` record → withheld → `failed`;
honest report, partial-failure recovery, and the `NOPE` probe all still `passed`; the format-vs-
literal discrimination holds; `GHOST_UNACKED_FAILURE_GATE=0` restores prior behaviour exactly.
**Live corpus: 0 records now read back `passed` WITH a failure reason (was 3).**

**Twelve mirror sites, four of which were not on the original list** — the hand-mirrored ladder in
the late backfill is DELETED (it now calls `resolve_turn_outcome`), and the flag also reaches the
lesson-outcome flush (a withheld PASS no longer ticks a lesson success), the selfhood diary,
`calibration.grade_turn_outcome` (falls to the graded 0.38 exec-failure path — "unverified and
something broke", not the hard 0.0 "checked and wrong"), and `_record_episode_safe` (whose label
feeds the playbook-lesson LLM and gates `search_recoveries`). 16 guards mutation-checked, each
turning a test red.

**`passed` + `structural failure` — root cause found: `iter_trajectories` was a SECOND WRITER
nobody counted.** The overlay only ever FILLED an empty `failure_reason`, never cleared one; the
writer's clear touched the in-process object only. Disk (`failed`/`structural`) + overlay (`passed`)
= a state combination no writer ever wrote. Closed at both ends. **Read-path overlays deserve the
same invariant audit as write paths.**

✅ **Latent shadowing hazard — FIXED same day (operator: "fix the 2 loose ends").** The kwarg that
shadowed the module-level `unacknowledged_total_failure` is renamed **`unacked_total_failure`** in
all three consumers (`resolve_turn_outcome`, `calibration.grade_turn_outcome`,
`agent._turn_outcome_label`); the public function keeps its name. Suite unchanged at 11119, which
is the correct signature for a pure rename.
⚠ **The rename itself nearly shipped a worse version of the same defect** — worth remembering
because it will recur: **BSD `sed` silently ignored the `\b` word-boundary pattern**, so the three
DEFINITIONS were renamed while two BODY references still read the old name (`agent.py:7441`,
`calibration.py:286`). Neither module imports that function at module level, so both were live
`NameError`s on the outcome path, and a green-looking mechanical edit would have shipped them.
Caught by grepping for leftover references rather than trusting the substitution. Note also that
`unacknowledged_total_failure_for_trajectory` shares the old name as a PREFIX — a careless
replace corrupts it.

✅ **The laundered record — REPAIRED same day.** `f78c8b33` now reads
`failed` / `structural failure` via an appended `source="operator_overlay"` correction (the original
write stays as audit history). The shape rule INDEPENDENTLY agrees with the corrected label
(`unacked=True`), so the record now reads the way the rule would have labelled it live.
**Corpus-wide sweep after the repair: 1492 trajectories, 0 reading `passed` with a failure_reason,
0 `passed`-but-unacknowledged-total-failure.** No laundered positive remains in the training corpus.

**Live confirmation on the deployed code (2 probe turns, 2026-08-04 14:39 and 15:18).** Both times
the agent answered HONESTLY (a skill-playbook lesson visibly fires in the thinking trace), so the
WITHHOLDING path did not run — expected at 8-in-1491. What ran, and is confirmed live: rule 2b
correctly STANDS DOWN on an acknowledged total failure (both tools failed, reply acknowledged →
`unacked_total=False`), while the counterfactual on that same live trajectory with the reply
replaced by `"0"` gives `unacked_total=True` → `failed`. The withholding direction is verified
against the real `f78c8b33` record and this counterfactual, but is still UNEXERCISED by a live turn.
⚠ **Both probe turns were labelled FAILED by the VERIFIER, not by the shape rule** —
`LATE REFUTED (100%): Constraint violation: the user requested 'just the number'`. An honest,
correct, helpful reply is refuted for failing a constraint the ENVIRONMENT made unsatisfiable.
That is the 07-31 incentive problem arriving through the verifier instead of the outcome ladder.
n=2, deliberately NOT acted on — but if it recurs in the corpus the fix is verifier-side (an
unsatisfiable constraint is not an agent violation), never another outcome-ladder change.

**Also found, not fixed (own round):** `_looks_like_tool_error` searches the WHOLE result for
`EXIT CODE: N`, so a nested banner inside a SUCCESSFUL `manage_projects` JSON payload marks the call
failed (live instance found). The shape rule is gated on the strike ledger specifically so it cannot
be hurt by this.

**Defect inside this round's own fix, caught by an existing test:** a first draft moved
`if cached is None: return` ABOVE the stashed-lesson flush, silently re-opening the 2026-07-26
lost-success-tick bug. `TestPassedFlushOrdering` caught it — the fifth consecutive round where a
regression test caught the fixer.

**Suite: 11119 passed / 14 skipped / 0 failures** (baseline 11046, +73). ⚠ NOT DEPLOYED — the live
agent still runs the 11:52 code; this lands on the next restart.

### 2026-08-04 (later) — AUDIT ROUND 2: the §4J backlog closed, and THREE of its own claims disproven on measurement

**Origin.** Operator: *"when you are done with all bugs from all the audits, bring everything in
shape for the GEPA project, if we need to reset and restart auditing, if we need new baselines, so
be it."* Four more fresh-eye agents over the triaged §4J backlog, disjoint file ownership (no VCS
here), every one told to write executable probes and re-derive the numbers this journal asserts.

**The headline is not what was fixed — it is that three §4J entries were WRONG, and only a probe
could tell.** §4J item 4 said `failure_distill`'s cluster gate is "structurally unreachable": the
live state file shows **three clusters fired on 2026-08-04**, and the real numbers are 19 corpus
records / 37% unattributed (not 69%) / 6 groups / 2 at threshold. §4J item 5 said 2 positive
compression deltas in 200 runs: there is **1**. And §4J's own fix for `escalated_overturn` —
recorded here as "now persisted" — persists into `VerifyResult.to_dict()`, which has **zero
production callers** (160 OVERTURNED lines in the log, 0 occurrences anywhere under
`$GHOST_HOME/system/`). A fix inside the audit was itself the audit's signature defect class.

**═══ FIXED ═══**

**The bench measured a different system than production (§4J item 1, escalation axis).** Detail in
§4J. Two independent derivations of the overturn rate, 84% and 80.8%, and the trap that a naive
grep reads 89% because OVERTURNED is a WARNING mirrored to a second logger. Shipped an
`EscalatingChatClient`, arm-qualified metrics (`fpr_raw_judge` XOR `fpr_escalated` — no bare `fpr`
key survives), `high_stakes` threaded through `run_trials` so the CONFIRM direction is exercised
too, and pre-08-04 bundles rendering `arm UNRECORDED` rather than being back-dated. En route it
found the verifier's 90 s main-leg bound being SWALLOWED by `_bounded_fallback_kwargs` — the
escalated arm would have measured a more patient adjudicator than production.

**The cheap judge's CONFIRMED direction (§4J item 2).** The sharper live finding: **0 of 130** cheap
verdicts sit below the 0.7 consumption gate (0.9×51, 1.0×79), so that gate filters NOTHING and every
cheap CONFIRMED was consumed unconditionally while every REFUTED got a strong-model check.
`_escalate_confirm` fires only on high-stakes turns and — deliberately asymmetric — does not flip to
REFUTED: it keeps CONFIRMED and caps confidence at 0.6, below every consumption gate. No fabricated
PASSED, no manufactured failure. `GHOST_VERIFY_ESCALATE_CONFIRM=0`, default ON.

**The §4F watch metric had nowhere durable to live.** The trajectory record CANNOT carry it: on the
streamed path `_record_turn_trajectory` runs at `agent.py:17147` and the verdict is spawned at
17336 — the line is on disk before the verdict exists, and the web UI always streams. Stamping it
anyway yields a field present on non-streamed turns and absent on streamed ones, with "no
escalation" and "verdict landed late" indistinguishable. Instead: an append-only ledger at
`$GHOST_HOME/system/verifier/escalations.jsonl`, written where escalation RESOLVES, recording BOTH
outcomes plus `unavailable` (a call was spent) — a ledger of numerators is why "84%" needed hand-run
archaeology. Identity passed DOWN the call chain, never read off the context. Surfaced in
learning-health per (route, kind); averaging routes would cancel 84% against 0%.

**Code-path refute escalation — MEASURED NEGATIVE, shipped default-OFF.** All 7 live cheap code
refutes replayed twice on the 35B: **14/14 upheld**. Mechanism, not luck: claim-path false refutes
are derived-fact failures ("49152 bytes → 48 KB", "latest PostgreSQL is 18.4") needing world
knowledge the 4B judge lacks, while every live code refute was a constraint/completeness check it
gets right. Also 2 of 14 replays returned empty at the 2048-token budget. `GHOST_VERIFY_ESCALATE_CODE_REFUTE=1` to enable.

**GEPA fix re-audit: 7 of 8 held under constructed failure scenarios; 1 did not.** The
`experiment_*` counters resolved LAZILY, on the first trajectory that reached them — so on a corpus
where nothing survives the earlier drops, "filter never ran" and "0 excluded" printed identically
(today's mine drops 219 of 490 post-era records). Now eager, plus `experiment_filter_errors` for a
filter that raises. Three NEW defects: `optimize_tool_descriptions.py` had **no resolution refusal
at all** — the only one of the three runners without it, and its private tier is the coarsest;
`run_gepa.py`'s A/B gate ran on a 30 s default where a timeout scores FAILED, racing the arms
UNEQUALLY because the longer-output arm is the slower one (measured 32.2 s cold → now 360 s); and a
leaked monkeypatch in the new test helper.

**Miner/runner flag collision.** `mine_tool_fixtures --min-fixtures` counted ALL fixtures while its
only consumer counts POSITIVES under the same name and default — the miner would have declared
"ready" and ATOMICALLY OVERWRITTEN the live pool at ~71 positives. `--min-positives` added, plus the
runner's resolution verdict printed before a run is started.

**Log legibility.** `RoutingTask.CLASSIFY_FAILURE` lowercases to "classify failure", so seven
healthy INFO dispatches read as failures in the live stream — the operator asked. Fixed with a
display-label map (`tag failure-dimension`); the routing string stays canonical because tests, docs
and timeout lines assert on it, and the 2026-07-12 "echo the REAL task" rule still governs anything
unmapped.

**Also:** mastery was inflatable by duplicate re-rolls (`regex_parse` had 10 of 10 recent outcomes
flagged duplicate — one real run from mastering on pure re-rolls); `failure_distill` now stamps
`_last_run` so a barren pass and a pass that never ran are distinguishable; `Trajectory.cluster` is
None on 1488/1488 and now WARNS rather than silently reporting a constant.

**═══ THE ARTIFACT VERDICT ═══**

**`planning.decompose` is contaminated as a measured win. Keep the file, de-record the claim.**
Re-scored offline on the same hash-stable 28-example private tier, both arms temp 0:

| metric | seed | promoted | |
|---|---|---|---|
| RECALL (the metric that promoted it) | 0.429 | **0.857** | reproduces the 0.45→0.80 promotion |
| TOKEN F1 (the metric that ships) | **0.500** | 0.071 | rejected |

Median output **111 distinct tokens against a 32-token gold** (seed: 35, matched) — it is the
verbosity optimum the old objective paid for. Consumer is dark (no `--use-planning`), so this is
correctness-of-record, not a live regression. Artifact untouched, sha `d47efe9c…`. Do not revive;
re-promote only against a bench that grades plan QUALITY, since neither metric does. **Related:**
post-fix that signature trains on 96 examples that are **100% reflection-sourced**
(`planning_output` is populated on reflection trajectories and nowhere else, 157/157) — "19%
contamination removed" understates the shipped state.

**═══ NEW BASELINE ═══**

`ablation_out/watch-4f/t1-20260804/` REPLACES `t0/`, which is no longer diffable: the case pool
changed, the bench gained an escalation arm, the GEPA metric went recall → token F1, and the
calibration keys were re-scoped at the 08-02 epoch fix. Contains artifact sha256s, a learning-health
snapshot, the supply state, and a README naming each incomparability.

**THE GATE BASELINE — measured 2026-08-04 12:02 (~75 min, live endpoints, both directions):
`private_incumbent_balanced = 0.766`.** 29 cases → 220 trials (166 refute / 54 non-refute), step
**0.0093**, arm `judge+escalation` (worker `100.83.184.117:8088` → main `127.0.0.1:8088`). Raw mean
0.722; non-refute 0.852 / refute 0.680. 97 CONFIRMED · 120 REFUTED · 3 UNCERTAIN · **0 skipped**;
71 refute overturns; 17 high-stakes trials, 0 confirms withheld. Templates adjudicate `ef79421a`
(tuned) / enumerate `c7b9e47a` (baseline) / pool `6666b453`. Stored at
`$GHOST_HOME/system/eval/verifier_incumbent_baseline.json` and copied into the t1 bundle.
**This supersedes 0.840 and 0.756** — the same templates re-read across sessions on the 4-case
tier, whose ±0.08 spread WAS the 0.0833 step rather than a change in the system. *Sanity check:*
71 of ~191 cheap refutes overturned = 37%, far below production's 84%, which is CORRECT because
166 of 220 trials are deliberately corrupted — a bench rate near production's would mean the fault
injection had stopped working.
⚠ **Three fields are absent from this recording** (added after launch, deliberately not re-run):
`route_health` (cheap-leg fall-through cannot be ruled out — route failures logged at *debug* in
the version that ran, so "zero LLM-call-failed warnings" is not a bound), `confirm_eligible`
("0 withheld" is ambiguous between "strong model agreed" and "never invoked" — the exact ambiguity
that counter exists for), and per-trial rows. Re-record via `--incumbent-only` when a reason
appears; all three are recorded by the current code.
⚠ **Known-and-deliberate, flagged not silently changed:** the CONFIRM direction cannot move
`optimize_verifier`'s gate metric — `_trial_score` is verdict-only and a withheld confirm changes
only confidence — so under `--escalate gate` it costs one main-model call per eligible trial for
zero gate signal. Kept for production fidelity.
**Also measured:** the cheap judge never expresses uncertainty — confidence distribution 1.0 (79) /
0.95 (32) / 0.9 (19) across 130 verdicts, which is why the ≥0.7 consumption gate filters nothing.

**Suite: 11038 passed / 14 skipped / 0 failures.** Deployed by operator restart 11:52:21; clean
boot, `system ready`, no tracebacks. Post-restart defaults: claim-refute escalation ON, confirm ON,
code-refute OFF, ledger ON. The ledger file appears on the FIRST escalation after a boot — its
absence right now is expected, not a broken instrument.

### 2026-08-04 — SELF-LEARNING STACK AUDIT: six parallel reviews, ~100 findings, and four subsystems that had never run

**Origin.** Operator: *"code review all self learning stack of the agent including GEPA … spawn
agents that will do fresh-eye reviews, fix all issues, repeat as many times needed … your endgoal
is a bug free agent"*, unattended.

Six reviewers in parallel — GEPA/optim, reflection+distill, calibration/confidence, idle loops,
memory learning surfaces, router/PRM/eval — each told to read §3/§4 first (so deliberate decisions
are not reported as bugs), write EXECUTABLE PROBES rather than trust docstrings, spot-check numeric
claims, and hunt this project's signature defect class: things that are built, tested, and never
actually run. Full finding list, fixed and unfixed, is §4J.

**The result that matters:** the single highest-value defect was not in new code. It was
`type(x).__module__.startswith("ghost_agent")` — a mock-guard used at six sites that is **always
False in production** (`python -m src.ghost_agent.main` → `src.ghost_agent.*`) and **always True
under test** (`PYTHONPATH=src` → `ghost_agent.*`). Four subsystems §3 recorded as `live` had
therefore never executed on the live agent, for weeks, with green tests the whole time. Live proof
took one `ls`: `failure_distill_state.json` had never been created.

That is the clearest example yet of why "the suite is green" and "the feature works" are different
claims — and it was found only because the reviewers were required to check behaviour against the
LIVE stores rather than against the code.

**Fixed:** the module guard (+ a fence test), self-play contaminating the production calibration
corpus and competence prior, infra outages being charged to the agent as genuine self-play
failures, episode outcome labels (96.5% wrong live), competence observation counts destroyed on
first write, redaction corrupting the live corpus (33/41 IP hits were not hosts), a quarantined
lesson creating a permanently un-learnable topic, and eight GEPA defects including a promotion path
that could DESTROY a better artifact and a recall-only metric that made verbosity optimal.

**Not fixed, recorded:** ~12 clusters in §4J, headlined by verify_bench measuring a materially
different system than production (which makes the currently-planned verifier round unsafe to run
as designed) and the complexity router being measurably good and 100% inert.

**Suite: 10767 passed / 14 skipped.** One self-inflicted break during the session (a scripted edit
put a continue at the wrong indent in `skills.py`) was caught by the targeted suite within a minute
— the tests did their job on the one occasion the author did not.

### 2026-08-05 — THE INSTRUMENT: live randomized arms + the risk governor that finally consumes the depth signal + tool-ontology analysis (3 features, 16+9+3 review findings fixed)

**Origin.** Operator asked for three high-impact ideas, then *"proceed with all 3 ... Make sure you
review your own changes using fresh-eye agents. be very careful"*. Three fresh-eye reviewers
(correctness/wiring, statistics/claims, data-integrity/tests) ran against the finished code. They
returned **28 findings, 8 of them critical, ALL past a green 10.6k suite** — including one defect
that one of my own fixes had created. Everything below is post-review.

**⚠ ALREADY LIVE.** launchd restarted prod mid-session (21:34 and 21:42), so the running agent
picked these changes up without an explicit deploy. Verified healthy: no tracebacks from the new
modules, and 6 real turns are stamped in the corpus. Kills: `GHOST_EXPERIMENTS=0` (whole
framework), `GHOST_RISK_STEER=0` (steer only), `GHOST_TRIAGE_RANKING=0` (reflection ranking) —
each needs a restart.

**1. `core/experiments.py` — live randomized arms. The instrument every parked item was waiting for.**
Each eligible request is assigned, blind and deterministically (`sha256(salt|kind|exp|req_id)`), to
one arm of each running experiment; the arm is stamped on the turn's trajectory; the outcomes that
already exist become a randomized comparison over REAL traffic. This is the earn-keep post-mortem's
pre-approved "change the instrument" route: no prod stop, no synthetic ceiling (B4 saturated twice),
no attribution ambiguity (the §4F watch has no control arm). Registry at
`$GHOST_HOME/system/experiments.json` (mtime-reloaded, malformed → built-in defaults). Read via
`introspect action='experiments'` or `scripts/experiment_report.py`.

**2. `core/risk.py` — the consumer for two measured-but-unused predictors.** §4H recorded that turn
DEPTH predicts failure (17.8% at step 1 → 60.6% at 12) and that the calibrated composite (AUC 0.727)
has NO behavioural consumer. Live half: a one-shot steer on deep+struggling turns whose third
instruction is the one the agent never takes alone — stop and report honestly. Offline half: §4H
item 2, reflection now ranks a bounded pool worst-first instead of taking the OLDEST failures.
**Backtested before trusting it:** fires on 101/1323 user turns (7.6%), first-fire clustered at
steps 6–8, and those turns fail at **0.543 vs a 0.277 baseline** — a targeting result, not an
effect. Whether it helps is the `risk_steer` experiment's job to answer, not this entry's.

**3. `optim/tool_ontology.py` + `scripts/tool_ontology_report.py` — is the toolbox carved right?**
Phase 2b optimizes description PROSE; the 0.772 ceiling check says the misses CLUSTER into pairs,
which is a boundary question prose cannot answer.
⚠ **The clustering did NOT reproduce (2026-08-05, §6).** On 67 held-out cases the 12 misses land on
**12 distinct pairs, none occurring twice** — so the premise this tool was built on is unsupported
on fresh data. The tool is still the right instrument; its motivating observation is now the thing
under test rather than a given. Too few misses to rule a boundary problem out either. Two read-only analyses: confusion classification
(bidirectional = merge/redraw, one-way = describe, no-tool = missing affordance, every verdict
tiered by an exact binomial symmetry test) and consecutive-n-gram macro mining. Proposes only —
promotion stays operator-gated. `--confusion-out` added to the Phase 2b runner (opt-in) so the
misses that the 08-03 run printed and discarded are now persisted.
**First live run:** `file_system` runs dominate — 789 pair-occurrences across **108 distinct turns**
at 0.72 cohesion; the 2/3/4-grams describe the same calls and would remove ~400 loop steps each. A
batch/multi-path `file_system` call is the highest-value macro on the board, and since depth drives
failure it attacks the failure RATE, not just latency.

**GEPA / §4F Phase 2b interaction — closed deliberately.** The steer mutates the prompt context, and
`GHOST_LLM_RECORD=1` is on for 2b supply, so a steered turn's later payloads would enter the fixture
corpus that the optimizer replays VERBATIM. Handled like the era filter: `tool_fixtures.py` excludes
turns with a mutated context by default, COUNTED as `experiment_context_excluded` (plus
`experiment_filter_unavailable` when the import fails, since "0 excluded" and "filter never ran" must
not look alike). `--include-experiment-context` overrides. `core.experiments.CONTEXT_MUTATING_KEYS`
is the extension point and carries the contract. Verified: miner unchanged, 180 fixtures / 62
positives, exclusions 0. **`optim/trainset.py` reads only user_request/final_response/cluster/tier —
the new `extra` keys cannot leak into GEPA examples.**

**═══ THE REVIEW FINDINGS (all fixed; each was invisible to the suite) ═══**

*Statistics — the instrument would have produced WRONG VERDICTS:*
- **Zero-width intervals.** The plain sample SD is 0 on constant input → the CS collapsed to a
  point → six Bernoulli observations "proved" a difference. Measured single-arm miscoverage
  **52% (p=0.5) / 82% (p=0.1)** at a nominal 5%. Fixed with a running-mean regularised variance
  (WSR's ¼ pseudo-observation generalised to arbitrary scale). Re-measured after the fix, from
  n=30: **2.5% / 3.3% / 9.2% / 1.8%** (Bernoulli .5/.3/.1, Normal). The p=0.1 row is the honest
  weak spot and is documented as such.
- **"Valid under continuous monitoring" was an overclaim** — the CS is ASYMPTOTIC, no
  finite-sample guarantee. Added `_MIN_VERDICT_N=30`/arm (numbers still shown, conclusion
  withheld) and the word asymptotic everywhere.
- **No multiplicity correction:** three metrics × "stop when any crosses" ran at ~15% vs nominal
  5%. α now splits ÷3 across metrics as well as ÷2 across arms.
- **Differential attrition, the sharpest trap and the LIKELY failure mode here:** `failure_rate`
  conditions on a POST-treatment variable, and the steer literally instructs "STOP. Report what is
  known" — which produces ungradeable turns. Simulated with ZERO true effect this manufactured a
  **14-point "improvement"**. Now the UNKNOWN rate is itself compared and `failure_rate` is flagged
  CONFOUNDED; `n_steps`/`duration_s` carry a permanent "mechanism, not outcome" annotation.
- **Balance alarm falsely accused the stamp 11.9% of the time** at its own minimum n (fixed ±20%
  rule vs a Binomial reality) → exact two-sided binomial tail, p<0.001. And it **died silently at
  n≥1024** (`2**n` overflows float64 → OverflowError → p=1.0) — now log-space. Same fix in
  `binomial_symmetry_p`.
- **`merge_or_redraw` fired on noise** — 2-vs-1 has symmetry p=1.000, and at n=57 with skewed tool
  usage pure noise produced a spurious merge directive **39.6%** of the time. Verdicts are now
  tiered significant/suggestive/insufficient with the exact p reported; thin pairs read "WATCH, do
  not act" (suppressing them entirely was my first fix and it was wrong — the pattern is still
  worth seeing).
- **`steps_collapsed` overstated savings 1.75–3.4×** (overlapping windows vs what a macro can
  actually replace) → non-overlapping counting; the live table was 1431→402 and is corrected in the
  doc with a footnote. Ranking also gained a per-turn cap: one 50-call grind session outranked a
  genuine 9-turn macro by 5.7×.
- **Two different "fidelity" numbers** were claimed to be the same; both are now reported.

*Wiring — the live path:*
- **Self-play/dream solver turns were being enrolled and steered.** `dream.py` calls `handle_chat`
  without a `request_id`, so `is_internal_request` (prefix test) missed them → ~50% got the
  treatment steer → this randomizes self-play outcomes, frontier scoring, and the **lesson
  keep/kill verdict** — and their context has no collector, so no arm is ever stamped: the
  experiment was perturbing a population it could not see. Now gated on collector-present +
  not-simulation + not-selfplay + not-read-only-memory. General rule: **a turn whose arm cannot be
  recorded must never be enrolled.**
- **The steer could contradict `force_final_response`** — a turn already told to write its final
  answer has tool calls dropped, so "run one small check" instructs a discarded action (and a
  dropped MUTATION surfaces a user-visible "not applied" note). Now withheld.
- **The futility mutual-exclusion claim was false in one direction.** Rather than suppress the
  more-specific breaker, the comment now states the actual (correct) policy: deference is one-way.

*Data integrity — and the defect MY OWN FIX created:*
- **The compliance bit was not request-scoped, and the arms ring made it worse.** A streamed turn
  writes its trajectory after the semaphore is released, so a context-attribute flag belongs to
  whichever request is running THEN. Both directions reproduced: a control turn stamped as steered
  (poisoning the triggered-only block AND dropping a clean GEPA fixture), and a steered turn
  stamped without its bit (defeating the GEPA isolation on the common path). Flags now live in the
  same req_id-keyed ring as the arms.
- **~13× dilution.** The trigger fires on 7.6% of turns; averaging over all traffic would have read
  "no difference detected yet" for ~4 months while the treatment was really changing behaviour.
  The report now renders a TRIGGERED-ONLY block and points the reader at it. Conditioning is
  legitimate: the gate is evaluated identically in both arms, only the action differs.
- **No zero-enrollment detector** — an empty report reassured identically whether the framework
  shipped 10 minutes ago or the stamp had been broken for weeks. Added stamp coverage
  (`N/M recorded user turns carry an arm`) to the header and a warning to the empty case.
- **`mine_tool_fixtures.py` destroyed the live GEPA artifact before its own gates ran** (verified:
  a wrong `--trajectories` took it 116 bytes → 0). Gates now run first; a failed mine is parked at
  `.notready` and the live file is left alone (`--force-write` overrides).
- **`tool_ontology_report.py` printed a CONCLUSION with no corpus** ("no recurring sequences") —
  this project's own "measured the corpus, not the signal" failure in miniature. Now exit 2 with
  an explicit error in `--json` too, and its GHOST_HOME fallback matches the miner's.
- `report_from_trajectories` materialised the whole corpus on the event loop → single streaming
  pass + `asyncio.to_thread` at the introspect site. `recent_samples` moved off the loop too.
- `seen_failures` silently changed meaning on an operator-facing line (pool, not quota):
  `summary()` now says "reflected 3 of 18 scanned".
- Registry now validates ARM names/counts (it is a file the agent itself can write, and arms ride
  into `extra` on every trajectory); null/NaN calibration composites are skipped rather than
  scoring **maximum** risk.

*The triage defect BOTH reviewers found independently:* mixing the calibrated score with the shape
proxy in one ranking measured **worse than either alone** (Kendall τ 0.605 vs 0.766 / 0.763), and on
the live corpus 7 of 8 joinable failures scored LOWER under the calibrated path — a real 12-step
disaster fell out of the reflection queue in favour of a milder turn with no sample. Now ONE scale
always: calibrated only when it covers the whole pool (~5% join today → shape in practice).

**Tests.** +99 across `test_experiments.py` (42), `test_risk_governor.py` (30), `test_tool_ontology.py`
(30), `test_experiment_wiring.py` (14 — includes the trivial-greeting fast-path trap that made an
earlier version pass vacuously). The third reviewer mutation-tested the suite; every surviving
mutation it found is now pinned, including a real coverage test (the old one *named* coverage and
never measured it, which is exactly how the 52% variance bug stayed green). One flaky test
(3.1% by design) fixed by pinning req_ids. **Suite: 10704 passed / 13 skipped.** One honest gap
recorded in-file: the `force_final_response` guard has no automated test, because every cheap way to
reach that state asserts the mock rather than the guard.

**Docs.** `docs/core/experiments.html`, `docs/core/risk.html`, `docs/algorithms/tool_ontology.html`
(all three carry the measured numbers AND the corrections), index + `self_improvement.md` updated.

**═══ ROUND 2: A FOURTH REVIEW OF THE FIXES THEMSELVES (same day) ═══**

Because one round-1 fix had CREATED a defect, the fixes got their own fresh-eye review. It found
12 more, **3 MAJOR — two of them inside round-1 fixes**. Worth internalising: *a fix is a change,
and changes are what this project's defect history is made of.*

- **The non-overlapping counter I added to stop OVERSTATING savings then UNDERSTATED them by
  49.6%** — one greedy cursor was shared across all n-gram sizes, so a match of sequence X
  consumed windows belonging to Y. Directional, too: same-tool runs claim the cursor first, so
  cross-tool sequences — the ones a macro proposal is actually for — were starved
  (`file_system → execute` read 45 against a true 110). Cursor is now per-sequence.
- **The evidence tier I added computed `p_sym`, printed it, and never read it** on the merge
  branch; `_SYMMETRY_ALPHA` was dead code. Measured against the noise model: a spurious "merge
  these two tools" directive fired **92.8% of the time at n=400**. Fixing it took FOUR attempts,
  each measured rather than reasoned about: (1) `observed > expected` — 98.5% spurious, because a
  comparison is not a test; (2) Poisson excess vs marginals — 42.8% at n=1000, the null failed to
  model that a miss can never land on the true tool; (3) leave-one-out marginals — power 95% but
  **90% false positives**, over-corrected; (4) shipped: full (conservative) marginals + Poisson
  excess + Bonferroni, plus a second route for the null's blind spot (a pair that IS ≥40% of all
  confusion cannot show "excess" because it *is* the marginals; noise never exceeded 34%).
  **Final: 0% spurious to n=1000, 2% at n=3000; power 64–97%.**
- **The arms ring was ALIASED into every shallow-copied context** (`subagent.py`, `dream.py` do
  `copy.copy(context)`), so a self-play cycle or delegate fan-out could evict a live user turn's
  entry before its streamed drain wrote the trajectory — recreating the exact loss the ring was
  added to prevent. Fixed by only ringing ENROLLED requests (background turns produce `{}`).
- `arm_for` was the only reader that skipped the ring, so a turn could take the control path while
  being RECORDED as treatment. `_regularised_sigma` did not reduce to WSR's ¼ for non-binary
  [0,1] data (anti-conservative by 2.5× for a clustered rate metric — latent, since `_METRICS` is
  the documented extension point). Plus: FIFO ring eviction, one malformed record disabling
  ranking for a whole pool, a learning-health corpus walk left on the event loop, and two
  reporting-consistency nits.

**Also verified by that round (worth recording as confirmed-good):** the log-space binomials are
exact to 4.4e-14 against `Fraction` and cost 2.7 ms at n=20000; both ring races behave; a
`mark_trigger` on an evicted id can never read as enrolled; `summarize_streaming` is byte-identical
to the two-pass form it replaced and genuinely single-pass; `compare_arms` really uses α/6 per arm;
`rank_for_triage` never drops or duplicates; the miner leaves a live artifact byte-identical when
its gates fail; and a full before/after stat of `system/` (362 files) confirmed the three scripts
write NOTHING to operator data.

**Monitoring added (operator request):** a boot line naming every live experiment and the steer
state; the control arm's "WOULD have steered, withheld" counterfactual promoted to INFO so both
arms are visible in the live stream; stamp coverage folded into `introspect action='learning'`;
and a one-shot verdict announcement into the autonomous-activity ledger at `notify` severity
(digest + push) — chosen because that ledger's two severities already encode the operator's
"auto-surface only actionable events" preference, and a decided verdict is the definition of
actionable. Persisted marker keys make it fire at most once per (experiment, scope, metric,
direction).

**═══ ROUND 3: THE MERGE GATE IS REMOVED, NOT FIXED (same day) ═══**

A fifth review re-derived the merge gate's claimed numbers with its own generator and they **did
not reproduce**. Worse: **both error rates move the WRONG way as the corpus grows** —
false positives 0.0% → 11.7% → 50.1% (n=400 → 3000 → 6000) while power fell 74.7% → 3.9% → 0.0%.
Three independent causes, each worth keeping as a lesson:

1. **The marginal null was BIASED, not noisy.** `obs/exp` for the hottest pair converged to a
   constant **1.20** under skewed tool usage instead of decaying to 1.0, because
   `P(pick=b | truth=a) = q_b/(1−q_a)` makes the miss-table column marginal `q_b·K` with K>1 —
   so the renormalisation was wrong, and wrongest for the busiest tools, i.e. exactly the hot
   pair. A constant multiplicative bias makes a Poisson p decay exponentially in n, so the gate
   eventually fires on pure noise with certainty.
2. **One condition ACCEPTED a null.** `p_sym >= alpha` requires FAILING TO REJECT exact 50/50 —
   an acceptance region that shrinks to nothing as n grows, so any genuinely two-way pair is
   rejected once there is enough data to see it. Measured on a fixed injected defect: c3 pass
   rate 95.4% at n=57 → 0.0% at n=10000.
3. **The Poisson tail underflowed to a constant 1.0** above λ≈745 (`math.exp(-lam)` → 0.0),
   switching the route off in silence at ~17k rows — and just BELOW the cliff it returned 0.0,
   i.e. falsely maximally significant.

**DECISION: the statistical adjudication of two-way pairs is REMOVED, not repaired.** Five gates,
five refutations by measurement. `poisson_excess_p`, `expected_pair_misses`, the dominance route
and the symmetry condition are deleted. A bidirectional pair is now reported with its raw shape
(both directions, share of all confusion) under `evidence="observed"`, with no tier that could be
mistaken for a mandate. The one-way `describe` test SURVIVES because it REJECTS a null rather than
accepting one — valid inference, and the only part of the design that held up across three reviews.
**This is a deliberate scope reduction.** The instrument's job is to surface candidates for a human
decision; adjudicating them was an addition that never earned its place, and dead-or-biased
statistical machinery is worse than none.

Same round, four more real defects — three of them in code from the previous round:
- **`mark_trigger` re-opened the ring eviction `_stash_arms` had just closed.** Sub-agents run the
  same turn loop on a shallow-copied (shared-ring) context, so 40 sub-turns evicted a live user
  turn whose streamed drain had not yet written its trajectory — recorded with no arm, the exact
  loss the ring exists to prevent.
- **An unwritable marker was a push-notification storm**: four notify-severity records per tick,
  **every hour, forever** (realistic trigger: a root-owned file under a UserName launchd daemon —
  this project already has a memory note for that failure). Now bounded to once per boot with a
  loud warning naming the path.
- **Mechanism-confounded metrics were being PUSHED as wins.** `n_steps`/`duration_s` move by
  construction for a stop-early treatment; the first tick measured 4 pushes, 2 of them
  "TREATMENT BETTER — ⚠ mechanism, not outcome". Confounded metrics no longer interrupt (they stay
  in the report).
- A marker containing a bare JSON string iterated into single CHARACTERS and wrote them back; a
  verdict could be marked announced with no ledger to deliver it; the learning-health corpus walk
  was left on the event loop while its sibling was deliberately threaded; and an unscoreable
  trajectory ranked as the SAFEST item in a worst-first queue without saying so.

**Two docstring claims failed their spot-check and are corrected in place:** the noise
hottest-pair share ("peaked at 0.343" — measured max 0.625 at n=57) and the savings overstatement
("1.75–3.4×" — measured 1.34× corpus-total after the cursor fix, median per-candidate ratio 1.00).
The per-sequence cursor fix itself was independently re-verified and reproduced exactly.

**═══ ROUNDS 4-5: the CUPED centring defect, and a transposition that would
have sent an operator at the wrong tool ═══**

Two more reviews, on the §4I router work and on the ontology removal. Both found
CRITICAL/MAJOR defects in code written the same day.

**CUPED narrowed the interval but never re-centred the estimate** (`compare_arms`).
`cuped_adjust` produces a series whose mean is `Ȳ_arm − θ(X̄_arm − X̄_pool)`, and that
RE-CENTRING IS THE VARIANCE REDUCTION — it removes the realised covariate imbalance from
the estimator. The code adopted the shrunken width and threw the correction away, leaving
the interval too narrow by `1/√(1−ρ²)` around an estimator whose variability had not
changed, while `variance_reduction` advertised the shortfall as a win. Measured with ZERO
true effect, monitored 30→300/arm: **12/300 false "TREATMENT BETTER/WORSE" verdicts (4.0%
on ONE metric, against a 5% budget for all three)**, growing with ρ² — i.e. dormant
exactly until Phase 2 says the covariate is worth having. Re-centring takes it to
**0/300**, re-verified here. The test asserting "the MEANS are untouched" had LOCKED THE
BUG IN, and the docstring plus `docs/core/experiments.html` both stated it as a safety
property; all three corrected.
Also from that round: the `min(raw, adjusted)` adoption rule really is the
better-of-two-draws effect (consistently signed, 25:2 across paired trials) but worth only
0.05–0.13pp — so it was replaced with unconditional adoption, which is simpler and honest.
In-sample θ turned out NOT to be a problem (coverage indistinguishable from baseline with
a useless or moderate covariate), which is the opposite of what I expected going in.

**The backtest's DISCRIMINATES gate had no uncertainty control.** A bare spread threshold
over point estimates said "DISCRIMINATES" on **perfectly FLAT ground truth 88.0% of the
time** at the default bucket size (99.8% when re-run as the corpus grows) — and the FLAT
branch is what stops §4I. It already computed the per-bucket confidence sequence and
ignored it. Now the best and worst buckets' intervals must be DISJOINT (Bonferroni across
buckets): measured **88–90% → 0.0%** across every cell, with a real separation still
detected.

**The ontology's merge detail transposed its two direction counts.** `tools` is sorted for
stable identity while `count` belongs to `truth→picked`, so whenever `truth > picked`
lexicographically the numbers printed against the wrong names — "browser lost 25 times"
when in fact `file_system` was right and `browser` stole it. An operator would rework the
tool that was CORRECT, and `--json` was equally blind. Each count now names its own
direction and the verdict carries `directions`.
Same round: verdict order depended on replay-row arrival order (61.5% of matrices reordered
under a shuffle — with a `top` cutoff the same corpus hid a different finding each run);
the "N verdicts above" sentence counted rows the render never showed; and
`OntologyVerdict.evidence` defaulted to `"significant"`, so any future call site omitting it
would silently claim a rejected null. Four docstring claims were also measured and found
overstated, and are corrected in place.

**Also fixed this round:** `mark_trigger` re-opened the ring eviction `_stash_arms` had just
closed (sub-agents share the ring through `copy.copy`); an unwritable marker meant four
pushes per tick FOREVER, now once per boot with a warning; mechanism-confounded metrics were
being PUSHED as wins; a NaN produced a zero-width interval; `cuped_adjust`'s `zip` truncated
silently; CUPED's sample gate was pooled rather than per-arm; and `router_confidence` is
NOT strictly pre-treatment across a conversation (the router sees the previous — possibly
treated — assistant message), which is now documented rather than claimed impossible.

**Suite after all rounds: 10718 passed / 13 skipped.**

**What to do next:** let it collect. No verdict is emitted below 30 turns/arm, and the triggered
block needs ~30 triggered turns per arm (~7.6% of traffic, so a few weeks). Read it with
`introspect action='experiments'`. If the 2b GEPA run wants a totally steer-free corpus, run it
before this collects much, or leave the default exclusion on and accept slightly slower supply.

### 2026-08-02 (later 2) — uCONSOLE UI: the web UI's face ported to the handheld, three QPainter faces DELETED, whole client restyled to glass + real bubble typography (deployed live)

**Origin.** Operator, after the voice work: *"the UI / UX and its 'face' need redesign, can we have a
'face' that looks like the webUI"* → then *"remove all three QPainter faces and implement the same
faces the web UI has"* → *"why do we split the screen if the left window is almost 100%
transparent?"* → *"do the rounded bubbles rewrite, and fix the padding"* → *"titles are way too big
… text doesn't look great"*. Five passes in one session, each driven by looking at a screenshot of
the real panel. Files: `interface/externals/clockwork_ghost/{client.py, webface.py, webface/,
chatlog.py, deploy.sh}`. **The device is the source of truth for what these look like — every claim
below was checked with a `grim` screenshot, not inferred.**

**1. The face is now the WEB UI's face, hosted in a QWebEngineView.** Rather than reimplementing
~2,700 lines of particle engine + GLSL in QPainter (a fork with a permanent maintenance tail), the
handheld runs the real `matrix_graph.js`: same thermal palette, all 8 forms, dive, pulses, error
flinches. `deploy.sh` re-copies the canonical `interface/static/matrix_graph.js` on EVERY deploy so
the two clients cannot drift. three.js r160 is vendored under `webface/vendor/` (module +
EffectComposer/RenderPass/UnrealBloomPass + deps) so the device needs no CDN at runtime.
**Feasibility was measured before building:** WebGL 1.0 is available (three falls back from WebGL2,
which is blocklisted on the V3D driver), and cost is purely FILL-RATE bound — 16 fps @1280×720,
35.7 @854×480, 50.9 @640×360, with busy-state + dive costing *nothing* extra. `GHOST_FACE_ZOOM=2.0`
shrinks the CSS viewport (browser upscales) and at 640×329 also trips matrix_graph's own
`IS_MOBILE` query (`max-height:600px`) → 120 nodes instead of 250, halving its O(n²) proximity loop.
That mobile profile exists for exactly this class of GPU.

**2. All three QPainter faces REMOVED** (Iris / Smoke Oracle / MoE Network — the device-only
`~/bin/face/` package, which never existed in the repo). `◈` now cycles FORMS, not renderers.
**A measurement that contradicted our own docs:** the docs claimed hidden faces cost zero CPU via
`showEvent`/`hideEvent` gating. On the deployed build the client sat at **61% of a core**; explicitly
pausing their timers changed nothing (60.8%); *removing* them dropped the client below the 3%
reporting threshold. The perf note was wrong — measure, don't trust it.

**3. Glass restyle.** The face moved from the right half to the **background of the whole window**
and every panel became tinted glass over it. Widget-over-webview compositing DOES work here, but
only with the face as an explicitly-positioned child plus `raise_()` on a
`WA_TranslucentBackground` overlay (the window is a fixed-size frameless kiosk, so no resize
handling). **The cyan accent was dropped**: the face's ring runs blue→violet→crimson with
essentially no green channel, so `#7be0ff` sat outside that range and made chrome and face look like
two different applications. Accent is now violet `#c9a6ff` from the face's own mid-palette, with warm
sand `#ffc08a` for the operator against its crimson core. Qt cannot do backdrop-blur, so readability
comes from tint alone — heavier fill behind text than behind chips.

**4. The left/right split is gone** (operator's observation: a nearly-invisible panel was still
reserving half the screen). The transcript container is fully invisible and spans the full width;
messages are free-floating bubbles, **operator right / agent left**, capped at 56% so the face stays
visible down the middle (the web UI caps at 35% for the same stated reason — to clear the sphere).
Input keeps bottom-left, action chips bottom-right; they share one row now.

**5. Bubbles rewritten as real widgets** (`chatlog.py`: QScrollArea + one QLabel per message) after
the table version proved unfixable — **`QTextDocument` supports no `border-radius` at all**, and its
table cells take a fixed percentage width instead of hugging content. Widgets give true notched
corners (16/4/16/16, web-UI parity), accent rails, and content hugging.
> **THE trap, and it cost two failed attempts.** A word-wrapped QLabel reports a *small*
> `sizeHint().width()` **regardless of `maximumWidth`**, so any size policy that defers to the hint
> (Maximum, Preferred) leaves every bubble a narrow ribbon. My first fix assumed a pre-layout width
> and raised the cap — the redeployed screenshot came back byte-identical, because **the cap was
> never the binding constraint**. `_Bubble.fit()` measures the text with wrap OFF, then pins
> `minimum == maximum` width to `min(natural, cap)` so the layout has no choice.

**6. Typography — the actual reason "text doesn't look great".** Two causes. (a) Markdown arrives as
bare `<h2>/<p>/<ol>` and **Qt applies its OWN heading scale on top** (`<h1>`≈2×, `<h2>`≈1.5× base),
so a briefing's headings rendered enormous; QLabel has no document-stylesheet hook, so
`style_markup()` injects sizing inline per tag, headings only slightly above body size. (b) The real
one: **everything was monospace, including prose.** The web UI splits this deliberately — agent
messages inherit the sans body font and only `.message.user` overrides to mono. Now prose is
DejaVu/Liberation Sans 20px; mono is reserved for the operator's own messages, inline code and code
blocks. Also styled: paragraph spacing + 148% line-height, list indents, inline-code chips, code-block
backgrounds, dimmed blockquotes with a violet rail, table header/column padding.

**Cost of the look:** translucent widgets must be re-blended as the webview repaints beneath them, so
the client runs ~39-49% of a core with QtWebEngine ~29-32% (load ~2.5 of 4 cores; RAM ~284 MB client
+ ~233 MB webengine). `GHOST_FACE_ZOOM` is the lever if it ever feels sluggish.

**Device/deploy facts worth keeping** (all in `deploy.sh` + `docs/interfaces/clockwork_ghost.html`):
the LIVE client is `~/bin/client.py`, NOT `~/clockwork_ghost/` (which does not exist on the device);
the launcher sources the `~/gui_env` venv (system python3 has no httpx — always compile-check with
the venv one); **QtWebEngineWidgets must be imported BEFORE QApplication exists** or Qt refuses to
build the view and the face silently degrades to blank; **ES modules do NOT load over `file://`** —
Chromium refuses with *no console error at all*, so the face is served from a loopback HTTP thread;
screenshot with `grim` after exporting `XDG_RUNTIME_DIR`/`WAYLAND_DISPLAY` (scrot renders black);
and driving the restart via `ssh host "bash -s" <<EOS` avoids the documented pkill self-match (the
pattern never appears in the remote cmdline). **Not verified remotely:** typing into the client —
the device has no `wtype`/`ydotool`/`xdotool`, so bubble/markdown rendering was verified by driving
the real `ChatLog` with representative messages, not by a live turn.

### 2026-08-02 (later) — AUDIO: the voice loop was dead at BOTH ends; re-pointed to local nodes + long-form transcription ingest SHIPPED (interface needs a restart)

**Origin.** Operator: "the worker model is gemma 4-e4b, it supports audio input, we haven't used it —
what can we do with it?" Live probe of nova found `"modalities":{"vision":false,"audio":false}` — the
projector wasn't loaded. Operator reloaded llama-server with `-mm
gemma-4-E4B-it-mmproj-BF16.gguf`; verified end-to-end the same session, then built three features.

**Measured on the live node (all constant across 3 orders of magnitude):** **25.0 audio tokens per
second** (3.2 s → 83 tok; 38.9 s → 974; 377 s → 9,435). **No encoder window limit** — codewords planted
at the start, middle AND very end of a 6.3-minute clip were all recovered, plus a faithful full
transcript, so the "~30 s window" caution in the earlier survey was WRONG for this stack; chunking is a
CONTEXT-BUDGET concern only. Throughput ~11× real-time (6.3 min in 33.6 s); longer clips are cheaper per
second (fixed thinking-token overhead amortises). Per-slot ceiling: `--ctx-size 131072 / -np 4` =
**32,768 tok/slot** ≈ 21 min of audio per request. `-np 4` also means audio need not block nova's other
duties — the older "nova is a slot, not capacity" framing was wrong by 4×.

**TWO TRAPS, both measured, both encoded in code + tests:**
1. **The empty-response trap.** Gemma 4 emits thinking blocks its chat template STRIPS, so too small a
   `max_tokens` returns **empty content with `finish_reason="length"`** — no error, just nothing (256 →
   `''`; 1024 → correct). Both call sites now RAISE on that exact shape: a silent `""` would have
   auto-sent an empty prompt to the agent (STT) or deleted a whole window from a transcript (ingest).
2. **Dialect.** Audio requires the OpenAI `input_audio` part; the `audio_url` data-URI shape
   `tools/vision.py` uses for images is rejected with `400 unsupported content[].type`. Deliberately NOT
   a copy of the vision payload builder. Both pinned by regression tests.

**THE REAL FINDING — the voice loop was dead at BOTH ends, and had been quietly deleted rather than
fixed.** `/api/stt` + `/api/tts` proxied to `PI_VOICE_URL`; its default `raspberrypi.local` does not
resolve and the launcher's override `http://disorder:8000` resolves on the tailnet but **listens on
nothing**. Because the buttons were therefore no-ops, both UI controls were removed as "unused" —
`#tts-toggle-btn` on 07-28, `#mic-btn` on 08-01 (the day before). A dead backend was diagnosed as an
unwanted feature. Classic silent-inoperative-subsystem, with the twist that the *cleanup* erased the
evidence.

**Shipped (all local, zero egress — the only speech path compatible with the no-keyed-API rule):**
- **`interface/voice.py` (new).** STT = ffmpeg transcode (any browser container → 16 kHz mono PCM, temp
  FILES not pipes: a WAV on a non-seekable pipe carries a placeholder length some decoders reject) +
  nova. TTS = macOS `say` (184 voices, no new model). `VoiceError` carries its own HTTP status so a
  client-side problem never reports as a 502. Text reaches `say` via `-f <file>`, never argv/shell.
  `PI_VOICE_URL` removed entirely; `server.py` proxies rewritten. Wire contract UNCHANGED (`{"text":…}`
  / audio bytes), so the browser needed no protocol changes.
- **Mic restored — in the INPUT AREA, not the header.** The header was trimmed to six controls for a
  verified single-row mobile layout; putting the mic back there would have undone it. Holding the mic
  now ALSO enables spoken replies for that turn and performs the autoplay-unlock gesture the removed
  toggle used to provide (iOS needs the silent-buffer trick, not just `resume()`); typing turns speech
  back off (`sendTypedMessage()`). So voice-in→voice-out with **no new header control**, and
  `#tts-toggle-btn` stays removed with its `if (ttsToggleBtn)` guard intact so re-adding it still
  composes. Cache-bust 8.7→8.8 (app.js + matrix_graph together).
- **`memory/audio_ingest.py` (new) — long-form ingest, sibling of `pdf_ingest`.** ~12-min windows
  (`-ss` before `-i` so seeking a 3-hour file is cheap), overlapping by 15 s so a sentence spanning a
  seam survives. Breadcrumb = **timestamp range** (`[talk.mp3] [12:00–24:00]`), the audio analogue of
  TOC breadcrumbs, stamped into the EMBEDDED text so a retrieved passage is CITABLE to the moment.
  Per-sentence offsets deliberately NOT claimed (the model returns text, not an alignment). One bad
  window is skipped into `stats.errors`, never fatal — same policy pdf_ingest applies to a bad page.
  Routed in `tool_gain_knowledge` BEFORE the plain-text branch (a `.wav` used to decode as
  replacement-char noise); video containers included (ffmpeg takes the audio track — a recorded talk is
  usually an .mp4). **Live: a 6:17 recording → 3 windows (`0:00–3:00`, `2:45–5:45`, `5:30–6:17`) → 8
  chunks, start/middle/end markers all recovered.** A 45-min talk is four windows.

**Two defects found by reviewing my own changes (both would have shipped green):**
1. **The 100 MB ingest byte-cap sits BEFORE the audio branch** — so every realistic conference-talk
   video (700 MB–1.3 GB) was refused at the door with "Split it into chunks first", advice that is
   meaningless for a recording. The cap guards RAM for text/PDF, but audio is never resident (one
   window's WAV at a time), so audio/video is now EXEMPT and the operative bound is DURATION
   (`GHOST_AUDIO_MAX_S`). The primary use case failed before this fix.
2. **`stats.seconds` summed window LENGTHS, which overlap** — a 6:17 file reported as "6.8 min
   transcribed". Now reports timeline COVERAGE. Caught by a test I wrote expecting the honest number.

**GEPA / §4F Phase-2b interference — CHECKED, negligible (measured, not assumed).** The interface work
is invisible (separate process, no LLM calls, no tool defs). The one prompt-visible change is
`knowledge_base`'s description (audio/video now advertised — otherwise the capability is undiscoverable
by the model). Against the in-era corpus (`ts ≥ 2026-07-31T19:15`): **372 records carry the old
description, 3 carry the new** (switch at `2026-08-02T16:46:38Z`), and **`knowledge_base` was chosen 0
times in the entire in-era corpus** — the mined tools are file_system 139 / execute 86 /
manage_projects 55, none of whose descriptions changed. The miner keeps full payloads and the adapter
swaps the CANDIDATE description at eval time, so a non-target tool's text is ambient context, not label
noise. No amendment to the mining plan needed. **One risk to carry forward:**
`registry._apply_tuned_descriptions` replaces a description WHOLESALE, so if
`tool_description.knowledge_base.json` is ever promoted it would silently delete the audio steer — the
capability would keep working while the model stopped knowing it exists. No `tool_description.*`
artifacts exist yet (only `planning.decompose` + `verifier.*`). Also: the tools block changed → one KV
stable-prefix re-prefill, already paid.

**Suite: 10,485 passed / 13 skipped / 0 failed.** New: `tests/test_interface_voice.py` (18),
`tests/test_audio_ingest.py` (21). `test_tts_passes_through_pi_content_type` REPLACED — the Pi
content-type passthrough it guarded is gone by design, not by regression. Two pinned registry contract
strings were restored verbatim after I broke them (the audio steer went into a separate sentence rather
than weakening the guard). Docs: `docs/interfaces/voice_server.html` rewritten (was documenting the
retired Whisper/Piper service), `docs/interfaces/web_server.html` endpoints + the removal notes,
`docs/memory/vector.html` audio-ingest section. Two `test_learning_health` entropy tests failed in one
full run and not the next, pass in isolation, and are not triggered by the new files —
**pre-existing order-dependent flakiness, unrelated.**

**POST-DEPLOY — two more defects, both found only by RUNNING it. Neither was reachable by any test I
had written, and the second is a repeat of a documented lesson:**

1. **`can't find variable: updateActivityIcon` — dormant code is not dead code.** Restoring `#mic-btn`
   revived ~130 lines that had sat behind `if (micBtn)` since 08-01, and one call in them had ROTTED:
   `updateActivityIcon` was deleted along with the center-stage `#activity-icon` on 2026-07-29, but
   **two call sites survived**. Site 2 was the mic path (broke STT immediately). **Site 1 was live the
   whole time** — in the file-upload success path, where the ReferenceError was swallowed by the
   enclosing `catch` and reported as `Upload Error: …` **after an upload that had actually succeeded**.
   A 4-day-old user-visible bug, found only because reviving neighbouring code made me audit the block.
   Both calls removed (the turn-status line owns icons now); guarded by a test that extracts every
   helper called inside the push-to-talk block and asserts it is defined — **and I verified the guard
   FAILS when the bug is reintroduced** (my first sanity check injected the defect outside the scanned
   block and "passed", which would have shipped a guard that guards nothing).
2. **`STT ERROR: HTTP 503` from Safari — the launchd PATH trap, AGAIN.** `_run_binary` 503s when
   `shutil.which()` misses, and **launchd hands a service a minimal PATH (`/usr/bin:/bin:/usr/sbin:/sbin`)
   that excludes Homebrew**. `say` is `/usr/bin/say` so TTS worked; `ffmpeg`/`ffprobe` are
   `/opt/homebrew/bin/*` so every STT request failed — **works from a shell, 503s under the daemon**,
   which is exactly why it survived all my testing. Not Safari-specific at all; Safari is just where it
   was tried. Fixed with `resolve_binary()` (explicit `GHOST_<NAME>_BIN` → PATH → known prefixes) in
   BOTH `interface/voice.py` and `memory/audio_ingest.py` — **the ingest path had the identical latent
   bug** (the agent is also a LaunchDaemon), unhit only because no recording had been ingested yet. The
   error text now names launchd and the override instead of the misleading "not installed". Verified by
   re-running the full round trip under `env -i PATH=/usr/bin:/bin`, with a **Safari-shaped MP4/AAC**
   upload rather than Chrome's WebM. Same defect family as the node binary that was invisible under
   launchd PATH — worth treating as a standing checklist item for anything a daemon shells out to.

**⚠ DEPLOY:** the agent side is live; the interface was restarted once and **needs one more restart**
for the two post-deploy fixes (`start-ghost-client.sh`). Cache-bust is now **8.9** (app.js +
matrix_graph together). `PI_VOICE_URL` in that launcher is inert and can be dropped. New env knobs, all
defaulted: `GHOST_AUDIO_NODE_URL` (tailnet IP — a dotless/mDNS name is what stranded the last backend),
`GHOST_STT_MAX_SECONDS`, `GHOST_STT_MAX_TOKENS`, `GHOST_TTS_VOICE`, `GHOST_TTS_RATE`,
`GHOST_AUDIO_WINDOW_S`, `GHOST_AUDIO_WINDOW_OVERLAP_S`, `GHOST_AUDIO_MAX_S`, `GHOST_AUDIO_MAX_TOKENS`,
plus `GHOST_FFMPEG_BIN` / `GHOST_FFPROBE_BIN` / `GHOST_SAY_BIN` absolute-path overrides.

### 2026-08-02 — 300 hourly warnings were measuring the CORPUS, not the signal: calibration epochs + delivered-Brier search SHIPPED (not yet deployed)

**Operator: "the agent has 300+ occurrences of `calibration: REJECTED the probability map` — is this
normal?"** Cadence: yes. Condition: no. ~291 lines of each message across the two logs = ~175 refits,
one pair per refit, ~1/hour — exactly `--calib-refit-cooldown 3600`. No runaway, no double-fire. But
`refit=ok` 107× → `map_rejected_inverted` 141×, flipping permanently at n≈1405 and never recovering,
and the live params carried `map_status: rejected_inverted`, `brier_raw 0.0542` vs
`brier_base_rate 0.0365`. **The guard fired correctly for 26 days while its diagnosis was wrong.**

**Root cause — three era shifts pooled into one fit.** `calibration.jsonl` is append-only and
`DEFAULT_MAX_HISTORY=4000` exceeded the whole 1709-row file, so every refit read all of it:

| era | n | mean competence | mean outcome | labels |
|---|---|---|---|---|
| 07-07 → 07-11 | 209 | 0.757 | 0.952 | binary |
| 07-12 → 07-19 | 691 | 0.828 | 0.964 | binary |
| 07-20 → 07-27 | 398 | 0.867 | 0.939 | binary |
| 07-28 → 08-02 | 411 | 0.898 | 0.838 | **graded** |

The LABEL changed 07-27 (binary {0,1} base 0.955 → graded base 0.855 — a redefinition, not a decline);
the FEATURES changed the same day (effort/entropy observed only from then, so 1298/1709 rows collapse
to competence alone under `_composite_for` and outvote the informative ones 3:1); and the competence
prior WARMS UP out of cold-start shrinkage. Pooled, **the score rises exactly as the labels fall** — a
Simpson's paradox that forces a negative Platt slope. Pooled: AUC 0.530, slope −0.077, map REJECTED,
Brier 0.0542 vs base 0.0365. Current epoch only: **AUC 0.722, slope +1.917, map applied, Brier 0.0244
vs base 0.0255 (−4.6%)**. Every window from 150 to 1500 fits cleanly; only the full history fails.
The oldest 209 rows alone flip it.

**Fix 1 — corpus epochs.** `CURRENT_EPOCH="2026-07-27.graded"`; every row carries an `epoch`; untagged
legacy rows get theirs DERIVED from `ts` (`epoch_for_ts`), never defaulted to current. `fit` reads one
epoch; so do `brier_score`/`ece`/`reliability_table`/`stats`/telemetry. Retro-negatives inherit the
CLOSING turn's epoch (they reuse its features — stamping them current would smuggle an old feature set
in with a negative label). **A label-scheme or feature change must bump `CURRENT_EPOCH`; that bump is
not bookkeeping, it is what keeps the fit valid.** Expect quiet until the new epoch clears the floor.

**Fix 2 — search the objective the pipeline delivers.** The grid scored candidates on RAW Brier while
the pipeline ships a Platt-mapped one: the raw-best point (0.054164, slope −0.077 → REJECTED) beat the
runner-up (0.054204, slope +0.388 → ACCEPTED) by **4e-5**. A coin flip in the 4th decimal decided
whether the agent got a probability map, with 321/396 grid points positive-slope. Now scored on the
delivered Brier, ties broken toward the simpler model (`_BRIER_TIE_TOL=1e-4`, compared on INTEGER grid
indices — `0.1+0.2 != 0.3` would otherwise rank equal-complexity points by rounding).

**The trap fix 2 opened — and why a tolerance can't close it.** Scoring on the delivered Brier removes
the level penalty that used to suppress useless columns, so the grid buys `w_entropy=0.2` for a 4.9e-5
in-sample gain on pure noise. Measured over 1000 noise corpora at n=40…800, that gain is
**heavy-tailed (median 0, tail ~1e-3) and does NOT shrink as 1/n** — so any tolerance wide enough to
absorb the tail also eats a real feature (at 1e-3 the live effort weight loses 4 of its 4.9 points).
The fix is an EVIDENCE gate, not a margin: `_MIN_SEPARATION_SIGMAS=2.5` requires
|mean(ok)−mean(bad)| ≥ 2.5 SE before a weight leaves zero. False-admits 4.9% of noise corpora (2.0σ:
8.8%); live effort separates at 3.64σ, entropy at 0.63σ → correctly pinned. This is the module's own
"a feature must VARY and SEPARATE" rule, finally applied in the fit rather than only in the report.

**Self-review pass found 5 more defects, all self-inflicted, all one shape** — a new filter applied to
SOME readers, leaving the rest describing a population that no longer exists (the "half-migrated
sanitized view" class):
1. **The gate rejected the PERFECT feature.** Zero within-class variance → SE 0 → returned 0.0σ. But
   constant-within-class and different-between-class is the best signal obtainable, not an undecidable
   one. Now `inf` when the means differ, `0.0` only when they don't (a truly constant column).
2. **Rendered ratios mixed scopes.** Epoch-scoped numerator over whole-file denominator printed
   `entropy observed on 418/1709 samples (77.3%)` — a fraction contradicting its own percentage.
3. **`_feature_health` contradicted the fit in adjacent lines.** Its verdict used a raw `abs(sep)<0.02`
   delta with no notion of noise, so entropy read `[live]` (0.0421 > 0.02) while the fit pinned it
   (0.63σ < 2.5σ), and the `features: N/4 live` headline counted a feature the fit refuses to weight.
   Now uses the fit's own σ gate. Live headline corrected **2/4 → 1/4 live**.
4. **Effort was judged over rows that never measured it** (n=497 vs 394) — the stand-in-averaging bug
   fixed for entropy on 2026-07-27 and never generalised. Now `_OBSERVED_FLAG`-driven for both.
5. **`outcome_pos/neg` counted only EXACT 1.0/0.0**, so under graded labels the line showed 155+/7- of
   541 rows — 70% invisible. Split at 0.5 like every gate in calibration.py; verifier-checked anchors
   kept alongside.

**Does this touch §4F/GEPA? No code coupling** — `optim/` has zero references to calibration or
confidence, `grade_turn_outcome` is untouched (so the graded-label trend the watch tracks is
bit-identical), and §4F's "threshold-aware calibration" (Phase 3 item 4a) is the verifier LOGIT PROBE
blend in `verifier.py`, a different subsystem. **One measurement interaction — see §4F caveat (g).**

**Tests:** `tests/test_calibration_epoch_and_objective.py` (39) + 2 rewritten and 2 new in
`test_learning_health.py`. Suite **10484 passed / 13 skipped / 0 failed**. Writing the tests caught two
further self-inflicted bugs: `epoch_for_ts("not-a-timestamp")` PROMOTED garbage to the current epoch
(`"n" > "2"` lexicographically — now shape-checked, and the failure direction matters: promotion is
the contamination the field exists to prevent), and `entropy_learnable` had silently stopped mirroring
the fit gate. Docs: `docs/core/calibration.html` §Corpus epochs + §Selecting on the delivered Brier.

**DEPLOYED + LIVE-VERIFIED 2026-08-02 20:37.** Functional tests on :8000, in order:
`/api/health` ok → a real turn recorded a sample carrying `"epoch": "2026-07-27.graded"` (deployed
record path) → live telemetry scoped correctly (546 in-epoch, 1168 excluded; effort 3.64σ live,
entropy 0.61σ pinned, competence 0.06σ dead; `features: 1/4 live`) → **first post-deploy refit at
20:37:48: `refit=ok threshold=0.83 w_entropy=0.00 lam=0.00 brier=0.02 n=546 excluded=1168`**, params
`map_status=applied platt_a=1.902957 brier 0.024188 vs base 0.025329` (beats by 4.5%), **and the
warning pair did NOT fire** (counts frozen at 116/117 since deploy). Hot-swap confirmed on the next
turn: `C=0.94 threshold=0.82` (raw, pre-refit) → `C=0.88 threshold=0.83` (Platt-mapped, post-refit),
matching `sigmoid(1.903·0.914 + 0.219) = 0.876` by hand. The prediction made read-only beforehand
matched the live fit exactly on every field.

**Two traps for whoever verifies a refit next.** (1) The idle window is bounded BOTH ways —
`900 < idle_secs <= 3600`. Stretches of 66m and 82m appear in the idle-cycle history and would NOT
fire; the agent needs a quiet 15–60 min, and unrelated traffic (5 requests in 6 min on the evening of
08-02) resets the clock. Budget for the wait; do not "help" by writing `calibration_params.json`
externally — the file would be indistinguishable from a real refit and would prove nothing. (2) This
log file carries NO DATE, only `HH:MM:SS`, and spans many days. Grepping `20:3[5-9]:` pulled refit
lines with n=1440/1119/1069 from previous days and briefly looked like several agents fighting over
one corpus. **Read this log in FILE ORDER (`grep -n … | tail`), never by time-of-day match.**

**⚠ THE SCORE HAS NO BEHAVIOURAL CONSUMER — the fix is instrument repair, not behaviour change.**
Traced post-deploy (operator: "does this mess with GEPA?"). `below_threshold` has exactly three call
sites: the dual-solver arbiter gate (`metacog.py:359`) and two counter/log-level sites
(`agent.py:11921`, `:16706`). The arbiter returns at `if not self.arbiter_enabled` BEFORE reading the
confidence, and live config confirms `toggle._METACOG_ARBITER_ENABLED = False` — the §3 module
constant, not the `--metacog-disable-arbiter` flag (which is False). So moving C 0.94→0.88 and τ
0.82→0.83 changed NO agent behaviour. Good news for §4F (there is no path from confidence to any
watch metric because there is no path from confidence to anything); sobering otherwise — this is the
"advisory not load-bearing / confidence logged-only" row of the §3 table, still true.

**Measured whether to make it load-bearing. VERDICT: mid-turn steer NOT WORTH BUILDING; use it
OFFLINE.** Read-only over 342 labelled trajectories carrying `tool_calls` (281 passed / 61 failed):

| still running at step k | 1 | 4 | 6 | 8 | 12 |
|---|---|---|---|---|---|
| eventual failure rate | 17.8% | 35.6% | 42.3% | 52.0% | 60.6% |
| AUC(effort@k) | 0.500 | 0.654 | 0.632 | 0.576 | 0.640 |

1. **Mid-turn, `effort_component` IS the spin term and nothing else.** At fixed k every prefix has
   exactly k calls, so sprawl is CONSTANT and only the longest same-tool run varies. Proven:
   `AUC(effort@k) == AUC(spin@k)` to three decimals at k=2..6, and effort@k has exactly k distinct
   values. Anything gated on mid-turn confidence is gated on repetition.
2. **The loop-breaker already catches that, more precisely** (same action + target + result, vs merely
   same tool NAME): by k=6 it fires on 40/71 turns at precision 0.53 / recall 0.70 of eventual
   failures; by k=8, 0.57 / 0.81. A confidence-gated steer re-derives an existing rule in
   probabilistic clothing.
3. **The signal is weakest where it would be most useful** — peaks 0.654 at k=4, decays to 0.576 by
   k=8, because the population still running by then is uniformly struggling.
4. **Not leakage, but weak.** Feared spin was error-retry in disguise; it is not — spin>=3 turns had
   an errored call 42% of the time vs 58% for spin<3 (reverse of the feared direction). On a
   leakage-free slice (no errored call in first 6, n=36) AUC(spin@6) = 0.570. Real, but thin.

**What to do instead:** (a) **offline triage** — post-hoc AUC **0.727**, materially better than any
mid-turn value precisely because the dominant term is the final CALL COUNT, which only exists once the
turn ends. Rank trajectories for reflection / postmortem / self-play attention by calibrated p(bad);
those currently select on binary "failed" flags and a graded calibrated score is strictly more
informative, at zero live-path risk. (b) **The depth prior is free and needs no model at all** — a
turn still running at step 8 fails ~52% of the time. That is directly usable for budget/escalation
policy without any feature, fit, or calibration. **Do NOT re-arm the dual-solver arbiter to "use" the
better score**: it was retired (§3) because it sampled 2 completions, threw both away and dispatched
the original — 0 answer changes at dominant latency. A better gate does not fix that.

**Note for whoever reads the suite next:** separate in-flight work (still in TESTING, not yet
journaled — not this session's, and deliberately not described here) was editing the tree while this
landed. Two tests failed mid-session and passed later purely from that movement
(`test_tool_registry_negative_constraints` — a tool-description string; `test_interface_face_palette`
— the app.js/matrix_graph.js cache-bust pair). **Neither is calibration-related and neither is a real
defect**; both are transient states of someone else's edit in progress. A green suite here is a
snapshot of a tree that was still being written to — re-run before drawing conclusions from it.

### 2026-08-01 (later 9) — req 56221fad post-mortem: the verifier refuted its own instrumentation; constraint lifecycle + claim fairness SHIPPED + DEPLOYED

**Incident (operator: "what went wrong on the last one?").** The 22:05 request drew a 90% LATE
REFUTED whose three reasons were ALL artifacts of our own plumbing, verified against raw
recordings: (1) "fails constraint to start with 'What it means to BE ghost'" — the model DID open
its final turn with the phrase; multi-turn assembly prepended turn-2's analysis and buried it at
char 2253; (2) "does not confirm project update" — the reply's ✅ confirmations sat past the blunt
`claim[:2000]` cut, and the system-appended ⚠Unverified/INCOMPLETE footer contradicted them;
(3) "truncated response" — finish_reason=stop; the "truncation" was the claim cap plus a garbled
footer (hedge auto-scan flagged our own note's "I cannot confirm it works" as a 40% assumption and
the risk summary re-rendered it — a self-echo). Root poison: a project constraint captured 07-28
("Start with: …") replayed into EVERY request for 4 days after the work closed, polluting new
artifacts (the phrase jammed above file titles) and feeding a refute→follow-up-task→reopen loop
the operator found as a queue of 6 junk "Verifier follow-up" tasks. The one legitimate defect in
the incident turn: the model finalized on an unverified write ("The tool response said SUCCESS.
I'll trust it.") — that part of outcome=failed was earned.

**Shipped (4 clusters, all tests green: 10,404 passed / 13 skipped):**
- **Constraint lifecycle** (`memory/projects.py`, `tools/projects.py`): project DONE retires
  `metadata.constraints` → `constraints_retired` (deduped, capped 20, event-logged) on BOTH DONE
  paths — `update_project` AND the raw-SQL task-rollup (`_maybe_rollup_project_status`, the way
  projects normally finish; review catch — the first cut missed it). Refute-driven reopens do NOT
  resurrect; the user restating one re-arms it (duplicate-create merge drops it from the retired
  list); forks/clones inherit active+retired via `_rearm_inherited_constraints` (a RELEASED parent
  necessarily passed DONE, so copying only the active list inherited []); new
  `manage_projects action=constraint_retire` (payload = text | index | all; text beats index so a
  digit-text constraint can't be shadowed). `_constraint_list` normalizer: bare-string metadata
  wraps to a 1-element list instead of being char-shredded and destructively persisted.
- **Verifier claim fairness** (`core/verifier.py`, `core/agent.py`): `pack_claim` head+tail
  packing (1200 + explicit "~N chars omitted — NOT a truncated response" marker + tail) replaces
  `claim[:2000]` inside `verify_claim`; idempotent. `strip_system_notes`
  (`core/reply_smoothing.py`) removes finalize-appended notes (⚠Unverified / Plan check / risk
  summary / leading correction banner) from the judged claim, evidence `claim_text`,
  `verify_code_output` response, `verify_visual` claim, the FILE-ARTIFACT filename parse (review
  catch: on streamed turns the banner carries the PREVIOUS refute's filenames → false FILE-ARTIFACT
  refute cascade), and BOTH hedge-scan sites (kills the self-echo footer). Terminal-block-only
  stripping (blank-line-free, end-anchored) so a model-authored "Assumptions I made" section
  followed by real content fails open. **verify_bench unaffected BY CONSTRUCTION: max claim in the
  exact seed-0 97-trial set is 373 chars — pack_claim never fires; no re-baseline needed.**
- **Start-with enforcement on the ASSEMBLED reply** (`utils/constraints.py`, finalize): if an
  explicit format mandate is active and a later paragraph opens with the phrase, the pre-answer
  narration is hoisted off (fence-guarded, ≥40% keep ratio, fail-open), after smoothing and BEFORE
  the verifier gate. Parse is DELIBERATELY narrow (review catch: bare "START with the parser" is
  ordering guidance, and a misparse here deletes delivered text): only "Start with: X",
  quoted-phrase, or reply-noun forms parse.
- **Follow-up filer artifact filter** (`agent.py`): refute issues shaped like packaging artifacts
  (response/reply/answer-truncation wording, "internal system message", "system noise") no longer
  become project tasks (banner-only). Anchored so truncated-DELIVERABLE issues ("export.csv is
  truncated at 100 rows") still file (review catch).

**Review discipline held (streak: 8):** two read-only execution-verifying agents on the green
diff found 1 CRIT (rollup DONE path unretired — the incident's actual finish path), 4 MAJOR
(string-shred persistence; banner filenames in the FILE-ARTIFACT parse; over-broad truncation
filter; ordering-instruction misparse deleting reply text), and 5 minors — all fixed same session
except one accepted audit-trail cosmetic (case-variant dedupe keeps first spelling). Every catch
is pinned in `tests/test_constraint_lifecycle_and_claim_fairness.py` (44 tests).

**Live cleanup + deploy:** project 7b62e5e533d1's stale constraint retired (event-logged) and all
6 junk follow-up tasks closed with an honest post-mortem note (store then auto-rolled the project
to DONE — correct, its work was finished); deployed via listener kill 19750 → 36971, clean boot,
bounded probe returned exactly PROBE-OK-3117. Docs: verifier.html, projects.html,
uncertainty.html. **⚠ 4F-watch attribution note:** this cluster reduces false-refute incidence
mid-window from NON-verifier-optimization causes — added to the §4F caveat (e) confound list.

One-feature adversarial review (16 findings: 1 CRIT, 3 HIGH) + full repair; interface
tests 170 green incl. new `tests/test_interface_visualizer.py`; live Playwright sweep
9/10 + direct forensics for both PDF paths. **Root cause (CRIT): PDFs rendered into a
NESTED iframe inside a `sandbox`'d iframe — the plugins-sandbox flag is unclearable,
inherited, and blocks the PDF viewer in EVERY engine (desktop was as broken as the
phone); the sandbox on that same-origin frame was a security no-op** (real containment
= `#render-iframe-sandboxed`, allow-scripts only, unchanged). Shipped: unsandboxed main
frame + direct `src = blob` on desktop; iOS panel with open-in-new-tab link (iOS can't
render PDFs in ANY subframe — platform limit); central `resetRenderSurfaces()` before
every open (agent code/audio kept RUNNING while hidden; stale mermaid masqueraded as
the new diagram; PDF never reset zoom → magnified blank margin); `_showRenderError`
panel for bad CSV / missing vendors (were silent black); srcdoc + `_escAttr` replaces
`document.write` (raced close navigation, can't follow a PDF src); PDFs bypass the
image LRU (dedicated revoked URL — 100 multi-MB entries was a phone tab-kill); LRU
spares the on-screen blob; `/clear`/session-switch close the window before revoking;
drag/resize un-broken on landscape phones/uConsole (media-query `!important` beat the
inline drag styles) + on-screen clamp; PDF Download branch (was silent no-op) + real
image filenames; `/api/download` gains nosniff + attachment-forced HTML/XML/SVG (agent
HTML would render same-origin with the key). Docs: web_server.html "Visualizer
overhaul". Versions: app/matrix 8.3, style 5.1.

**Addendum (same session): input-box misplacement WITHOUT rotation (operator: happens
"when the reply is long enough").** Third producer of the stuck-layout symptom:
`syncBodyHeight` pinned `body.style.height = visualViewport.height` on EVERY vv event —
one stale final reading (more streaming = more events = higher odds) left the body
pinned short forever, footer floating mid-screen. Fixes (v8.4): the pin clears to the
CSS 100dvh steady state when vv ≈ innerHeight; and a **geometry WATCHDOG** (800ms tick)
verifies both the body pin and the keyboard translate against FRESH measurements,
correcting >32px drift — stop chasing event producers (rotation, scroll-dismiss,
staleness), make every stale state self-heal ≤1s. Playwright: injected stale body-pin
(input at y=406 — the exact symptom), stuck translate, and both combined, all healed
with ZERO events fired.

**Addendum 5 (CLI, two rounds): long replies "don't use the whole screen".**
Round 1 mis-read the complaint as the shred bug and shrank the Live tail (screen/3 —
the full-height region only repaints cleanly when started at the TOP row; every
turn-2+ starts mid-screen). Operator: "still doesn't use the whole screen" → the real
requirement was flow semantics: the reply should FILL the screen as it streams, like
normal terminal output, not a windowed view + settle reprint. **Shipped
`_StreamPrinter`: progressive block streaming** — each COMPLETED markdown block
(boundary = blank line at even code-fence parity, `_flush_point`) prints PERMANENTLY
into the scroll flow the moment it finishes (rich routes prints above an active Live);
only the in-progress block rides the small live tail; settle flushes the remainder —
NO reprint, scrollback holds exactly one clean copy. This reconciles the two hard
constraints: markdown can't re-render printed lines (⇒ only finished blocks print) and
a Live region can't exceed the screen (⇒ only the unfinished block lives there).
tmux capture-pane verified: mid-stream the pane is FULL and flowing (26/30 rows,
tail -f-like), prompt ends on the last row, transcript exact (0 missing / 0 dup
across two turns), fences render. `~/Data/AI/bin/ghost` is a SYMLINK to the repo file
(deploys itself). tests: 317 cli+interface green. notifications panel gained a ⌫ Clear-all
button (local-only — records are acked at delivery, the watermark prevents re-serving;
badge+list+dedupe reset); hold-to-talk mic REMOVED from the input row (unused; engine
stays `if (micBtn)`-guarded dormant, same pattern as the TTS toggle — chrome pin
updated deliberately); header icons regrouped: bell → logs → workspace/upload/download
(file ops contiguous) → face → zen (appearance last). Placeholder "Enter command or
message..." → "Enter message" (didn't fit iPhone portrait). Chain: notifications 6.8 →
workspace 7.0 → app/matrix 8.7. Live-verified: order, clear behavior, no overflow, no
page errors.

**Addendum 3 (v8.6): mobile delete-all-sessions was IMPOSSIBLE** — the rail's
unconditional `pointerleave` disarm fired on every tap's trailing pointerleave (touch
pointers are transient), so tap 2 of the armed confirm always RE-armed instead of
executing. Disarm is now mouse-only (`ev.pointerType === 'mouse'`); touch keeps the 4s
timeout as its "no". Live-verified via touch-tap + synthetic pointerleave (no real
deletion); pin `test_delete_all_disarm_is_mouse_only`. Chain: sessions 6.9 →
workspace 6.9 → app/matrix 8.6.

**Addendum 2 (v8.5): the v8.4 "vv≈innerHeight ⇒ no keyboard" test was WRONG in
standalone PWAs — innerHeight SHRINKS WITH the keyboard there (Safari-browser keeps it
at layout height), so keyboard-open read as no-keyboard → pin cleared → typing blind
behind the keyboard (operator repro).** Keyboard presence now = focused editable + vv
below this orientation's MAX-SEEN vv height by >80px (`_vvKeyboardOpen`; baseline
resets on rotation). Watchdog went bidirectional: heals a stale pin (input floating)
AND a missing pin (typing blind). Playwright (standalone-style emulation: viewport
shrink, width constant): pin applied + input visible at the keyboard edge; missed-event
re-pin ≤1s; restore on close. Lesson: innerHeight semantics differ per display-mode —
never use it to classify keyboard state; derive from the vv's own per-orientation max.

### 2026-08-01 (later 7) — req 65d8cf76 post-mortem: one dropped '<' became five strikes; parser heal + replay scrub + path-aware hints SHIPPED

**Incident (16:39, after the later-5 deploy — NOT caused by it):** the model emitted an
otherwise byte-perfect equals-dialect call as `<tool_call>function=knowledge_base>` — the
`\n<` between opener and function tag lost upstream (single-token drop at temp 1.0 or a
server stream swallow; the boundary recording can't distinguish). llama-server passed it
through as content; the fallback XML parser (whose job is exactly this straggler class per
the 07-31 design) missed it by one character → `no_function_tag`. The broken text was then
replayed VERBATIM as the assistant turn ("preserve the raw XML so it remembers"), so at
temp 0.6 the model copied its own broken shape on turns 2/3/5 (in-context imitation beat
four recovery hints — which were teaching the WRONG dialect for the native path anyway);
turn 4 stalled → structural=6 → abort, failed·0.46. Evidence:
`llm_recordings/2026-08-01.jsonl` req 65d8cf76eb814134 (turn-1 history had ZERO XML
examples; turn-2 payload shows the verbatim replay).

**Fixes (agent.py):**
- **Heal:** `<tool_call…>` directly followed by `function…=` gets its `<` restored —
  anchored to the opener so parameter bodies can't be rewritten. Recovers all five
  incident turns; a sweep of ~4,300 healthy recorded contents (07-31 + 08-01) fired ZERO
  false heals.
- **Replay scrub:** on an actual failed block (`system_parse_error` entry — NOT
  `parse_failure_reason` alone; "truncated" accompanies fully-executed calls), the failed
  tool-call XML in the replayed assistant message becomes a constant-size note. Covers
  `<tool_call`/`<tool`/`<function(_name)` shapes, skips backtick-quoted mentions,
  hard-bounded (first 2 regions → notes, rest deleted), callable replacement (no template
  injection).
- **Path-aware hints:** `no_function_tag`/generic/escape-hatch recovery messages now
  serve the equals dialect on the native path (`_tool_call_format_example`, keyed on
  `args.native_tools`) — attribute style stays on legacy. Both examples pinned to
  round-trip through our own fallback parser.
- **Review catches (2 read-only agents, execution-verified, all fixed same-session — the
  streak holds):** scrub-gate false positive on executed mutations (would have asked the
  model to re-run a completed mutation); history serializer rendered the synthetic
  `system_parse_error` entries as an imitable attribute-dialect call once the scrub
  removed the `already_inline` suppression (renderer now filters them); scrub missed
  heal-only shapes (`<tool>`, `<function_name`); degenerate thousand-block replies
  amplified ~5x through the scrub (now shrink); the lesion behind an UNCLOSED `<think>`
  was swallowed silently by `_THINK_UNCLOSED_RE` strip-to-EOS (lookahead now knows the
  lesioned opener); note-as-regex-template crash; attributed-opener and unquoted
  `<function_name=` lesion variants.

Tests: `test_parse_failure_recovery_fixes.py` (30, incl. the exact live payload) +
path-aware parametrization of `test_system_parse_error_messaging`. Suite 10343 green
(2 unrelated known flakes pass in isolation: interface dead-subscription prune,
memory-bus concurrent fanout). Docs: `docs/core/agent.html` new section. DEPLOYED:
listener 8511→16682, health ok, bounded live probe through the full turn loop returned
exactly PARSER-DEPLOY-OK.

**Watch:** `system_parse_error` fires on the native path should now be rare AND
self-recovering (one note + correct-dialect hint, no imitation loop). Any repeat of a
5-strike parse abort is news.

**Follow-up (same session, operator: "fix this properly"):** the remaining dual-dialect
nudge CLOSED — `_render_assistant_with_tool_calls` now takes `native=` (from
`args.native_tools` at its single call site) and renders replayed history calls in the
EQUALS dialect on the native path — i.e. exactly the bytes the model generated before the
server parsed them into structured calls; legacy path keeps attribute style. Both render
dialects pinned to round-trip through the fallback parser. Reviewer on the renderer change:
change itself CLEAR (36-case adversarial round-trip harness, zero regressions; rendered
shape byte-matches recorded emissions) but it flagged THREE MORE attribute-dialect leaks
riding the native path — all closed same-session:
- **SPECIALIST_SYSTEM_PROMPT (MAJOR, the surface the 07-31 header split missed):** two full
  attribute examples (fix-script + CDATA rule) on EVERY coding turn, no native gate — the
  prompt-block class the ablation showed corrupts stacked calls 8/8. Now a
  `{{TOOL_XML_GUIDANCE}}` slot: legacy keeps the original verbatim; native gets the same
  fix-script workflow with ZERO tool-call XML and a raw-values rule instead of CDATA (the
  template parses values raw — a CDATA wrapper would land verbatim inside the argument).
- **Cognitive-watchdog break_text:** the synthetic replan call (persisted + replayed) now
  speaks the active path's dialect.
- **GBNF grammar foot-gun:** `GHOST_TOOL_GRAMMAR=1` under `--native-tools` is refused with
  a loud warning (grammar hard-codes attribute dialect → three-way conflict at the sampler).
Native path is now one-dialect END TO END: system+specialist prompts teach no format,
hints teach equals, history shows equals, sever-injections speak equals. The bigger unwind
(structured tool_calls through the template, dropping the Qwen-Agent string translation)
stays out of scope pending a 07-31-style ablation — logged as a §4B candidate.

### 2026-08-01 (later 6) — web client → mobile app model (PWA + resume + web push), operator-requested

Operator: "make the web UI fully mobile compatible; requests must survive locking the
iPhone; iOS notifications must work — like the Claude/Gemini apps." Shipped + live-verified
(interface tests 153 green; Playwright 7/7 on iPhone viewport incl. reload-mid-stream
recovery; REAL push delivered to both live device subscriptions, sent:2).

- **Mobile layout.** Face-form menu: two-axis viewport clamp + max-height scroll (the
  right-anchored menu near the LEFT edge of the centered ≤480px header hung ~114px
  off-screen). **Safe-area trap, bit twice:** a `padding:` SHORTHAND in a header media
  query silently resets the base `padding-top: max(…, env(safe-area-inset-top))` — in a
  standalone (home-screen) PWA the header sat under the clock/Dynamic Island. Re-assert
  insets after EVERY header padding shorthand (≤480 portrait AND the ≤1280×720
  landscape/uConsole block, which needs left/right too — the island owns the left edge in
  landscape; max() keeps the uConsole unchanged).
- **In-flight turn survival (the Claude-app spine).** The proxy already buffered streams
  (task TTL counts from COMPLETION — long turns survive) and the agent persists replies
  into sessions; only the HANDLE died with the page. Now: `ghost_inflight_turn` in
  localStorage (taskId+sessionId+tabId+5s heartbeat) → boot/visibility/pageshow probe
  `GET /api/chat/task/{id}/state` (new) → reattach the stream, or poll `/api/turns`
  ("Ghost is still working…"), or ADOPT the session history wholesale (never append —
  merge_history doubling). `/clear` drops the handle; session-mismatched handles are
  discarded; a fresh other-tab heartbeat blocks hijack, a stale one allows adoption.
- **Web push (iOS-capable).** Prereqs: tailscale serve :8443 → :8080 (SW + APNs need a
  REAL cert; self-signed :8080 cannot), dynamic key-gated `/manifest.webmanifest` (link
  injected by server.py; start_url carries the key; nothing keyed in unauthed /static),
  SW registration gate `(!isIOS || isStandalonePWA)`. Transport:
  `interface/webpush_notify.py` + VAPID keypair (~/Data/AI/.ghost_vapid.json 0600) +
  subs store (0600 from first byte). Producers: (a) reply-ready via the ACK-GRACE
  contract — live JS POSTs `/api/chat/ack/{id}` after rendering; no ack in 12s
  (GHOST_PUSH_ACK_GRACE) + not user-cancelled → push with request preview ("stream
  drained" is NOT delivery: dead sockets buffer the tail); (b) notify-severity ledger →
  consumer `web-push`, ≤5/cycle, last-acked watermark (no mtime churn).
- **REVIEW CATCHES (6th straight session; 2 CRIT + 5 MAJ past 150 green tests AND a live
  Playwright pass):** (1) push was 100% DEAD — pywebpush cannot parse PEM
  (`Vapid.from_string` b64-decodes); fixed by passing a py_vapid Vapid INSTANCE; the
  mocked-pywebpush tests were green throughout → added a REAL-crypto test (fake only the
  HTTP session). (2) deterministic double-resume on the exact iOS bg/fg flow (boot +
  visibilitychange + legacy resumeOnVisible all fire; async guard ≠ mutex) → synchronous
  `resumeLatch` + `resuming&&isProcessing` bail. Plus: cancelled turns pushed "Reply
  ready" (→ `cancelled` flag), /clear resurrection, reconcile-adopt racing a live turn
  (re-check after every await), VAPID-rotation deadness (client re-binds), multi-tab
  hijack (heartbeat), watermark ack churn, ticker leak, subs-file mode window, probe
  tasks at shutdown.
- **Keyboard-offset rotation bug (operator repro: focus input → rotate → rotate back →
  input pinned at the TOP).** `--keyboard-height` is computed as `innerHeight − vv.height`
  — during rotation those two briefly describe DIFFERENT orientations (a bogus
  half-screen "keyboard"), and iOS dismisses the real keyboard mid-rotate so no later
  event corrected the translate. **First fix (v7.9: coherence guard + settled
  re-measures) FAILED on real Safari** — WebKit can keep the element focused after
  killing the keyboard AND report stale visualViewport geometry until the next user
  interaction, so any recompute-based repair builds on numbers that lie. Shipped fix
  (v8.0): DON'T measure at rotation — force the state: width-change/orientationchange ⇒
  blur the editable (the keyboard is gone anyway) + write 0 immediately + settled
  re-measure later; focusout ⇒ 0; retry-capped coherence guard (an uncapped one can
  freeze tracking forever under persistent width incoherence); 55% clamp. Lesson
  upgraded: on rotation don't repair derived values — make the underlying state
  deterministic and derive from that.
- **Cosmetic follow-ups (operator):** PWA/notification icons swapped to the ghost
  artwork (`sips` from ghost.jpg → icons/ 180/192/512, `?v=2` busting; installed PWAs
  keep their install-time icon — remove + re-add); status label `SYSTEM ONLINE` → `ONLINE`
  (wrapped to two lines fullscreen).
- **Operator setup (one-time, documented in docs/interfaces/web_server.html):** open
  `https://eva.taila2b1d.ts.net:8443/?key=…` in Safari → Add to Home Screen → open the
  installed app → tap once to grant notifications. Version chain ended the session at:
  style 5.0, app/matrix 8.2.

### 2026-08-01 (later 5) — §4 remaining-work sweep: Phase-2b miner+read-site BUILT, Phase-3 flip prereq test, §4E Tier 3 SHIPPED

Cleared every §4 item that was actionable without waiting on a clock or evidence gate. Suite
10282→10344 green; live-validated where a live surface existed. **Review-your-own-changes paid for
the FIFTH straight session: 2 fresh read-only agents on my own green-suite edits found 1 MAJOR +
3 MAJOR + minors — all fixed same-session (details below).**

- **§4F Phase 2b — tool-description fixture miner + registry read-site (both halves of the loop
  that were missing; the GEPA run itself still waits on supply per the run plan).**
  `optim/tool_fixtures.py` + `scripts/mine_tool_fixtures.py`: one streaming pass, era filter
  (local→UTC), structured-tool_calls-only, ground truth via `iter_trajectories()` overlay joined
  on request_id (SYSTEM sentinel excluded), honest-failure exclusion, `toolfx:<request_id>`
  tiering, light fixtures with day-file `source` pointers. Read-site:
  `registry._apply_tuned_descriptions` — artifact-only (`tool_description.<tool>.json`, no
  OptimizableSignature, scope fence untouched), `_TOOL_DESC_OVERRIDES` offline hook, per-tool
  validator + 20k **aggregate** inflation guard (all-or-nothing; per-tool caps sum to ~10× the
  real tools block), copy-on-write, one-scan-per-process under explicit GHOST_HOME only.
  **Live mine (scratchpad, not promoted): 111 fixtures (22 pos / 89 neg), 45 clean
  `<tool_response` previews, supply gate correctly says WAIT** (~2-3 days per plan; note the
  labeled corpus skews negative because clean-PASSED requires ZERO failed tool calls — the
  55 honest-failure exclusions are mostly passed turns). REVIEW CATCHES (miner agent, live-
  measured): result previews were 79% injected `<system_state_update>`/playbook text (head-slice
  → now marker-slice), unbounded by_session retention (~100KB/record → OOM in weeks → one-pass
  pending-pair index), same-request summarizer at ordinal+1 mispaired (→ kind+marker guard,
  lost-pair-never-mispair), CLI zero-positive hole, SYSTEM join landmine, test-isolation leak
  via the loader's ~/ghost_llamacpp fallback. Tests: test_tool_fixture_miner.py (18),
  test_tool_desc_readsite.py (14). Docs: self_improvement.md §Phase 2b.
- **§4F Phase 3 flip prereq — stable-prefix-hash regression test DONE**
  (`tests/test_stable_prefix_phase3.py`, 7): flags-on vs flags-off payload byte-identity (pinned
  first-user msg + system slot), in-request cross-turn pin stability, BoN-copies-not-mutates
  (direct `_adaptive_bon_final` drive: k gen calls on `list(messages)+[...]`, judge on cheap
  pool, live list untouched), source guards (no Phase-3 env read before `_compose_injection`;
  BoN call inside the `if not _do_repair:` region). LEARNED: the pinned block legitimately
  varies ACROSS requests (persona/steering reclassify per request) — the KV contract is
  within-request only; a cross-request byte-identity test is wrong by design.
- **§4E Tier 3 — task_reopened retro-negatives SHIPPED (the last pending tier).** Join key =
  new `tasks.closed_req_id` column (stamped from `request_id_context` on the transition INTO
  DONE; SYSTEM blanks; **reopen CONSUMES the stamp** so each closing turn is labeled at most
  once and a stale stamp can't mis-attribute a later SYSTEM re-close) + `req_id` field on
  CalibrationSample/record() (Tier-1 passes it; durable — survives restarts, no dependence on
  the 32-entry stash). TASK-level DONE→open fires `on_task_reopened` (project status
  irrelevant — ACTIVE-project revivals count; add_task project-reopens deliberately do NOT — no
  single closed task to blame) → `task_reopened` event + main.py hook →
  `CalibrationTracker.record_task_reopened_negative`: re-records the closing turn's OWN stored
  components (no-leakage) at `_TASK_REOPENED_GRADE` 0.15, idempotent, skip-if-already-negative.
  REVIEW CATCHES (Tier-3 agent): named-column pre-SELECT would hard-crash every update_task
  after a skipped best-effort migration (→ SELECT * + dict.get + column-presence gate);
  record() swallowed write failures while the hook logged success (→ record() returns bool);
  stale-stamp scenario (→ consume-on-reopen). Accepted limitations, documented: same-turn
  close→reopen records nothing (turn sample not yet written); multi-process double-fire
  possible (per-process locks; rare, bounded harm); dup-scan reads full history (rare event,
  ~1.2k rows). Tests: test_task_reopened_tier.py (18). Docs: core/calibration.html,
  memory/projects.html.
- **§4F parked item 8 (FORCE_COLOR color tests) — already fixed** by the earlier 2026-08-01
  session (`_no_color_env_leak` autouse fixture); verified green under FORCE_COLOR=3. Item
  CLOSED in the snapshot below.

### 2026-08-01 (later 4) — 4F watch T+3d reading (early, ~T+2d): HEALTHY — refutes are TRUE positives, FP cost absorbed by escalation

Read over the surviving log segment (07-31 ~21:02 deploy boot → 08-01 ~08:53 — conveniently ≈ the
post-honest-failure segment; caveats a-f §4F applied). **Verdict: HEALTHY, no intervention.**
- **Activation continuous:** 38 GEPA template loads = 19 boots × 2 — tuned verifier live on EVERY
  process across the deploy churn. Phase 3 flags verifiably absent from the live env (default-off
  holding); 0 BoN fires.
- **Volume:** 242 final verdicts (200 CONFIRMED / ~54 REFUTED-family / 1 UNCERTAIN).
- **The FPR watch (the pre-registered concern):** 64 cheap-judge refutes OVERTURNED on main-model
  escalation (≈49% of ~130 cheap refutes) — the tuned judge still over-refutes, consistent with
  the bench direction (0.31→0.385), BUT the cost lands exactly where the design puts it: ~5
  extra main-model calls/hour, nothing user-facing. **Facial validity 4/4:** every sampled
  SURVIVING refute is a true positive (unverified-availability claim; internal-state noise in a
  reply; a missed second price check after a required delay; and correctly refuting req 50855398 —
  the known-failed Slack-notify request from this morning's postmortem). The churn is the
  instrument catching real failures, not crying wolf.
- **Correction churn:** 51 corrections queued-to-surface across the segment (~21% of
  verdict-bearing turns) — high but tracking the true-positive refutes; 23 bounded auto-repair
  rounds; honest-failure late-upgrade fired exactly ONCE (rule is quiet, not distorting).
- **2b supply (co-clock):** 87 structured tool-call fixture responses TODAY by 08:53 alone —
  miner-ready volume lands ~08-02/03 as planned.
Next: T+7d (~08-06) verify_bench re-run vs t0 (the controlled, verifier-specific evidence) +
segmented outcome-label trend.

### 2026-08-01 (later 3) — Mini AI v3 retest PASSED + notify promise now persists on the project

**Retest (4 requests, 07:47–08:00): the v2 test shape completed clean — 7/7 tasks persisted
(decompose fix), design gate honored + needs-user Slack DM delivered in ~30s, final autoadvance
5 tasks/292s/zero strikes (v2: 1789s, 6/6 abort), deliverables verified on disk (parse + demo
runs exit-0). The new inline per-dir transport and JSON-recovery paths visibly fired in
production. Model weaknesses persist (1 think-loop abort, 3 malformed spec parses) — all
contained. NOTE: req 5299219b died 3.9s in via an external graceful SIGTERM at 07:50:44
(operator restart presumed — NOT the agent, NOT my deploy); that request carried the only
"notify me in slack" ask, which exposed the last gap:**

**Notify promise now persists on the PROJECT (core/notify_promise.py): captured into project
metadata at request start (before the first LLM call — an early death can't lose it), fired
when the project settles (DONE/FAILED) from both finalize twins (model-notified dedupe) and
main.py's on_project_done hook (idle completions; stands down while a foreground request is
active). Atomic consume via _atomic_metadata_update → exactly-once delivery across racing
paths; rate-limited attempts leave the flag for retry; 'project' phase = Slack delivery without
chat-banner double-render. Tests: tests/test_notify_promise.py (13, incl. request-52 replay);
docs: core/autonomous_activity.html.**

**Review pass (5th straight with real catches — 5 fixed): internal-turn settles (sched-/sub-)
would have STRANDED the promise (finalize gated the fire on non-internal; now only the backstop
and CAPTURE are internal-gated — a sub-agent prompt echoing "notify me" can't store a phantom
promise, but a cron completion still delivers); the streamed twin read live current_project_id
in the post-semaphore drain (could fire/consume ANOTHER project's promise — now uses the
_drain_pid snapshot like work_log); a failed ledger write after the atomic pop silently ate the
promise (now restored via setdefault); an active promise now suppresses the per-request
backstop (premature mid-project "Done —" + settle fire = guaranteed double ping); brittle test
pins relaxed. Residual (documented): idle-loop FAILED rollups deliver on the next foreground
touch; a mid-stream client disconnect defers to the same.**

**Ship wobble worth remembering: the first capture wiring did a function-local
`from ..utils.logging import request_id_context` inside handle_chat — Python scoping made the
name local for the WHOLE function, so the `request_id_context.set(req_id)` at turn start blew
up with UnboundLocalError → 172 test failures on the full run. The module-level import (line
~35) was already there; the local import was pure poison. Full suite re-run green (10,231
passed). The remaining 2 "failures" were the session shell's FORCE_COLOR=3 leaking into
utils/logging's IMPORT-TIME color constants (RESET/DIM/BOLD/_LEVEL_COLOR bake _ansi() results
at import) — not a code bug; TestAtomicPrint now pins the module attrs via monkeypatch so the
suite is immune to the invoking shell. DEPLOYED 2026-08-01 ~09:0x (idle-checked before the
kill — request 5299219b's lesson applied to our own deploy; agent healthy on respawn).**

### 2026-08-01 (later 2) — req 50855398 post-mortem (1789s abort): model struggled, FOUR agent defects turned it into a failed request

**Operator question: "model not smart enough, or agent not good enough?" Verdict: BOTH, with
distinct shares. MODEL (Qwen 35B): 5 thinking-loop aborts in one request, double-escaped JSON
spec content (data_generator.py emitted as ONE line of literal \n), two find==replace corrupted
edits, from-scratch backprop with wrong gradient indexing it could not debug in 700s, and
title-as-directory paths ("Mini AI v2/train.py") two requests running. AGENT: four containment
gaps let that struggle become a 6/6-strike abort with the promised Slack notification never
sent (verifier late-refuted the turn for exactly that). All four fixed + a fifth steering fix:**
- **F1 (the abort's direct cause): inline `-c` conversion ran scripts from SHARED /tmp** —
  script dir = sys.path[0], AHEAD of the 2026-07-02 PYTHONPATH=$PWD fix, so a stale
  /tmp/ai_model.py (left by an earlier scaffolding script) shadowed the project's file; strikes
  5+6 were both `ImportError … (/tmp/ai_model.py)`. Now each script gets its own empty
  /tmp/_ghost_inline_<id>/ dir — nothing to shadow (tools/execute.py).
- **F2: promised-notification BACKSTOP at finalize** (`_notify_promise_backstop`, agent.py):
  the finish-line guard stands down on forced finals — exactly when the promise matters. On any
  non-internal end without a notify_operator call, the agent writes the notify record itself,
  honestly labelled, req_id-stamped (no same-turn echo), 12/h-budgeted.
- **F3: double-escape containment**: file_system syntax feedback names the shape outright
  ("THE WHOLE FILE IS ONE LINE … DOUBLE-ESCAPED — REWRITE"), and the coding executor REPAIRS
  such content pre-write via bounded unescape (common escapes only, accepted ONLY if the result
  ast.parses). Live cost had been 3 blind executor retries + ~180s of interactive re-discovery.
- **F4: System 3 pivot fit inside its own 120s wall**: generator/evaluator now /no_think +
  enable_thinking=false + max_tokens 1500/700 — the incident pivot burned its full 120s in the
  think channel and ReadTimeout'd during the crisis it existed to solve.
- **F5: title-as-directory catch in _missing_file_message**: missing path with a dir prefix
  whose basename EXISTS → "you are ALREADY inside this project's workspace — use the bare
  relative path". (Second incident in two requests.)

**Tests: tests/test_req50855398_fixes.py (19) + updated execute-conversion pins; docs:
tools/execute.html, tools/file_system.html, core/coding_executor.html,
core/autonomous_activity.html, core/agent.html.**

**Review pass (fourth straight session with real catches — 6 findings): (1) three stale
autoconvert tests still pinned the flat /tmp path (suite would have gone red); (2) the backstop's
`had_failures` keyed on force_stop, which terminal-tool SUCCESSES also set — "stopped early
after repeated failures — Dream cycle complete." would have paged the operator; now strike-count
only; (3) the title-prefix hint recommended the bare basename even when the file lives in a
subdirectory — now names the MATCHED path; (4) the backstop never ran on STREAMED finals (web UI
always streams — the incident shape exactly); twin wiring added in the streamed post-work block;
(5) simulation/dream turns with "notify me" in synthetic prompts could page the operator —
is_simulation guard added; (6) an ERRORED notify_operator call suppressed the backstop — only a
non-error call counts as promise-kept.**

**Also from this cluster: the F4 no-think switches tripped the two source-inspection guards
that cap `enable_thinking: False` occurrences in agent.py at 1 (trivial fast-path) —
test_self_play_redesign + test_tts_adaptive_bon now allow ≤3 with EVERY occurrence still
required to sit in a /no_think-marked or trivial-path block (the main-tool-turn guard intent is
preserved; the System 3 pivot is a legitimate bounded side-call). Full suite 10,219 green;
DEPLOYED 2026-08-01 ~07:45 via plain kill (this deploy also shipped the pending Slack
server-side halves from the "later" session). The retest that validates this whole cluster is
the "later 3" entry above (Mini AI v3: 292s, zero strikes, 7/7 tasks).**

### 2026-08-01 (later) — Slack notification workflow review: pipeline sound, four operational defects fixed

**Reviewed the whole outbound path (activity ledger → on_notify push → /api/notifications
pending/ack → bot poller → owner DM) plus the bot's interactive half. The 2026-07-11/13-era
core holds: owner lock fail-closed, thread filtering, delivery-before-ack ordering, watermark
clamp, the wedge fixes. Four operational defects, all fixed + tested (172 green in the two
touched test files):**
- **.err was 11MB/unrotated, .log empty since Jul 11** (the §5A watch item): basicConfig(INFO)
  → stderr only, and httpx logged every poll (2 INFO lines/30s ≈ 5.8k/day) — the 07-13 wedge
  signature was buried in its own noise. Now: rotating file handler on the .log path (5MB×3,
  GHOST_SLACKBOT_LOG; empty disables for tests), stderr WARNING+, httpx/httpcore silenced,
  hourly `poller heartbeat: N poll(s), M delivered, K error(s)` line as the positive liveness
  signal.
- **Idle-ack churn**: unconditional ack every 30s → no-op POST + notify_consumers.json rewrite
  around the clock (mtime permanently fresh = the "stale mtime" diagnostic dead). Bot skips the
  ack only when the watermark EQUALS the last acked value (empty-but-advanced still acks — the
  07-13 wedge case stays pinned); server-side save_consumer_offset skips identical-value writes
  for ALL consumers (web-ui too). Poller reuses one persistent AsyncClient, rebuilt on error.
- **Thread re-upload clobber (data-loss class)**: build_thread_context re-uploaded every earlier
  attachment on EVERY thread message — re-POSTing the original over the agent's sandbox copy,
  silently clobbering agent edits ("fix this file" flows lost the fix). Bounded in-memory Slack
  file-id cache skips re-uploads; restart re-uploads once (pre-fix behaviour).
- **Readability**: human phase labels + stale-age suffix (_(2.1h ago)_) on re-served records;
  EMOJI_MAP refreshed with current request-path icons (🧠 🧪 🔗 🧭 🎯 🔒 📣 🐘).

**Also surfaced: `interface/slack_project_commands.py` is a complete, tested `/ghost project…`
slash-command router that NOTHING imports (built-but-unwired) — wiring needs an agent-API-backed
store context; logged, not built. Docs: docs/interfaces/slack_bot.html,
docs/core/autonomous_activity.html, .env.example. Deploy = kill the bot pid (KeepAlive
respawns); the 11MB .err truncated during the restart window.**

**Review pass (third session in a row where it paid): the ack-identity skip had quietly deleted
the system's only self-heal loop — a TRUNCATED ledger's stale watermark was echoed by /pending
forever and the skip suppressed the blind re-ack + clamp that used to converge it (permanent
silent wedge until bot restart). Fixed at the SERVER: /pending adopts read_since's shrunk-ledger
re-baseline and returns the healed EOF watermark (+ regression test with a real truncation).
Also from the review: a non-2xx ack is no longer recorded as acked (one agent 500 would have
become a permanently suppressed retry under the identity skip); enabled:false responses are
never acked (literal watermark 0 would reset the offset → full history replay later); the
upload cache got a 6h TTL (uploads are project-scoped and sweepable — an eternal cache entry
asserts files that no longer exist); the rotating log moved OFF the launchd .log (rotation
under launchd's open fd diverts stdout into a backup) to ghost-slack-bot.info.log; an
unwritable log path now warns loudly instead of going dark. NOTE for other consumers (web-ui):
mirror the bot's ack contract — ack only on 2xx, never ack enabled:false.**

### 2026-08-01 — Mini AI incident (req b7e516b9): five ledger-integrity defects fixed + data repaired

**The operator asked what went wrong with project management in request b7e516b9. Root-cause
chain: (1) task_decompose's dedup keyed on the generic "Implement:"/"Research:" colon-head, so
5 planned tasks silently persisted as 3 — the agent then spent turns updating PHANTOM
train.py/demo.py ids; (2) the agent built core.py out-of-band, the ledger stayed PENDING, and
a re-invoked autoadvance re-executed the task, APPENDING a second implementation into the
working file (SyntaxError line 379) → task FAILED → project auto-rolled FAILED; (3) the agent
rebuilt+verified core.py within minutes but the recorded "does NOT parse" evidence was never
re-checked, wedging every close attempt; (4) the final autoadvance on the FAILED project
reported "All tasks are complete — the project is done"; (5) the verifier CONFIRMED the false
completion at 95% because its evidence never contained the ledger.**

**All five fixed** (tests: `tests/test_project_ledger_integrity_fixes.py`, 25 new):
- `_feature_key` generic-head fix + `dropped_duplicates` reporting on decompose/create;
  `task_update` unknown ids return `valid_tasks` + instruction (tools/projects.py).
- Coding executor pre-write guards: `_py_append_guard` (duplicate top-level defs / second
  `__main__` / merged-unparseable → refused BEFORE the write), overwrite guard (never replace
  a parsing .py with non-parsing content), `ALREADY ON DISK` spec steer + verify-only spec
  (`files:[]` + verify) accepted by `_usable` (core/coding_executor.py).
- Stale-evidence live re-check: `task_update status=done` on a FAILED task with parse-shaped
  failure_reason re-runs the syntax check on the CURRENT file — passes become result evidence,
  failures refuse with the current diagnostic (`_recheck_stale_parse_failure`); DONE clears
  `failure_reason` (core/planning.py).
- `advance_many` returns `project_failed` (never `project_done`) on a FAILED project/ledger;
  autoadvance payload carries `failed_tasks` from the store; `task_next` lists `parked_tasks`
  with the revival path.
- `_project_ledger_evidence` (core/agent.py): manage_projects turns append a live ledger
  snapshot to the verifier's claim evidence, inside the 4000-char cap.

**Data repair (live DB, operator-approved): core.py task DONE with live-verified evidence
(stale reason cleared), train.py task added retroactively as DONE (decompose had eaten it),
deliverables registered (core.py, train.py, research brief), project rolled DONE. demo.py was
never persisted as a task and never built — add a task if still wanted. Docs:
docs/tools/projects.html, docs/core/{coding_executor,project_advancer,planning,verifier}.html.
DEPLOYED by the operator same morning (plain kill); live-validated by the Mini AI v3 retest
(see "later 3": 7/7 tasks persisted, NEEDS_USER honored, stale-evidence wedge gone).**

**Review pass (fresh read-only agent on my own edits — the 2026-07-27 discipline paid again,
6 real findings past a green suite, all fixed + pinned): (1) the live re-check counted
UNCHECKABLE files (no node binary, unknown ext, read error → `_syntax_feedback`'s "" is clean
AND unknown) as passes — would have force-closed tasks on manufactured evidence; now only
verifiable types count. (2) The verifier ledger block was dead on the two most incident-like
turn shapes: manage_projects-only turns (bookkeeping run-gate skipped → autoadvance outcomes
with `"stop_reason"` now count as substantive) and execute-ending turns (code branch dropped
claim_evidence → ledger now rides verify_code_output's output). (3) Ledger snapshot used only
current_project_id — now also resolves project ids named in the tool outputs. (4) Verify-only
spec + kill-shaped verify closed a task with ZERO checks — now refused with feedback. (5)
Parent-cascade DONE promotions kept stale failure_reason — cleared like the direct path. (6)
The recheck path regex's space-tolerant class swallowed "py_compile failed for core.py" as one
bogus path — space removed, fail-safe direction preserved.**

### 2026-07-31 (later 5) — The caption-lesson hypothesis was FALSE: nothing to scrub, and the retraction mechanism had already worked

**Asked to scrub the "stale caption-workflow lessons", I audited the store first and found
they do not exist. My own earlier claim — that hydrated freq≥7 lessons from the pinball
incident were teaching the describe_picture workflow and beating the prompt steer — was
inference from a `memory bus  Hydrated context for:` log line plus the observed behaviour. I
never checked the playbook. It was wrong.**

**What the audit actually found** (50-lesson playbook + chroma twins, read-only):
- ZERO lessons mention describe_picture-as-verification. Two vision-adjacent entries exist and
  neither teaches the caption workflow: idx 35 (freq 7) is about a vision-node OUTAGE and webp
  fallback ("re-invoke the primary tool after conversion"), idx 34 (freq 4) is about chaining
  navigate→screenshot→analyze after a TargetClosedError. Both are sound; neither is stale
  enough to delete, and deleting a still-useful lesson is worse than keeping it.
- The pinball-incident lessons are GONE — because req 66d64313 ended `LATE REFUTED (100%) …
  scrubbing this turn's lessons`. **The retraction mechanism did its job automatically at the
  time**, which is precisely why there was nothing left to scrub tonight. Only one low-value
  pinball entry survives (freq 1, a project-bookkeeping lesson from an unrelated turn).
- No ORPHANED vector twins: the 12 chroma hits for pinball/describe_picture are episodic user
  messages and one SITUATION doc whose playbook entry still exists. The twin-deletion path
  (`_delete_lesson_twin`) is holding.
- Also searched the class today's OWN fixes could have made stale — lessons teaching
  "one tool call per response" as a corruption workaround, which would now suppress the
  parallelism the native-header fix restored. None exist either.

**Nothing was deleted.** The false claim is struck through in the (later 2) entry and in the
`ui-verification-channels` memory rather than quietly edited away — a wrong causal story in a
durable record is worse than the bug it describes, because the next session inherits it.

**The transferable lesson** (same shape as the 2026-07-22 memory-substrate finding "validate
silent-memory bugs against live DB, not logs): a hydration log line proves lessons were
CONSIDERED, not what they SAID. Causal claims about memory content require reading the store.
The simpler explanation for the residual behaviour stands: on a BROAD "does it render
correctly?" question a caption is a reasonable answer, and the model's own priors favour the
familiar tool — which is why the moment-of-use TIP on describe_picture results (later 2) was
the right lever, and it is prompt-level, not memory-level.

### 2026-07-31 (later 4) — Closed both watch-items instead of leaving them as watch-items

The two things the honest-failure + corruption work left OPEN are now fixed, not merely
monitored:

**1. Skills-auto graduation guard (the risk the honest-failure rule created).** Since
"the tool broke and I said so" is now PASSED, a turn can be PASSED having accomplished
NOTHING — and the extractor keeps PASSED trajectories with ≥2 tool calls, so an honest error
report could have graduated as a "skill". A skill is a path you would REPLAY, so a PASSED
trajectory now qualifies only when ≥`min_tool_calls` of its calls actually SUCCEEDED.
Rejections are COUNTED (`ExtractionReport.rejected_no_successful_tools`), not dropped
silently — a rising number is the evidence that the outcome rule is admitting turns this
pipeline should keep ignoring. Guard imports the shared `outcome_heuristics.tool_call_failed`
(new public alias) rather than writing a second failure sniffer — duplicated copies of that
judgement are exactly how the corpus and the operator line came to disagree. Skill identities
untouched (sequence key still built from all tool names → no store re-keying). Verified
behaviourally: all-failed PASSED ×3 → 0 candidates / 3 rejections; genuinely successful ×3 →
1 candidate.

**2. Repair fire = tripwire, not noise.** The corruption's root cause is fixed and the probe
battery produced zero fires, so a fire now means a NOVEL shape. The log line says exactly that
("UNEXPECTED since the 2026-07-31 native-header fix … this is a NEW corruption shape") because
for months it read as background noise and was scrolled past, AND the event is filed in the
background-activity ledger (phase `native_tool_repair`, INFO — durable + queryable via
`introspect action='activity'`, without interrupting the operator over one repaired call).
The raw pre-repair snapshot is carried in the record's meta, since it is the only artifact of
whatever new shape fired.

Tests: +4 in test_skills_auto_extractor.py (incl. a pin that the shared sniffer is imported,
not redefined), +2 in test_native_tool_header.py. Docs: skill_acquisition.html new
"Honest-failure graduation guard", prompts.html tripwire paragraph.

### 2026-07-31 (later 3) — Honest-failure rule: a broken TOOL is no longer a failed TURN (operator decision)

**Headline: `resolve_turn_outcome` ranked structural execution failure above everything, so a
turn whose only tool call failed and whose answer HONESTLY reported it ("that file does not
exist") was recorded FAILED — with the verifier CONFIRMING at 100%. Rules 3/4 swapped: a
verifier PASS now outranks the structural signal.** Flagged as a design edge in the previous
entry (probe F2), decided by the operator, implemented here.

**Why it mattered:** that label fed the trajectory corpus (Reflector / PRM / skills-auto) and
the operator's log line — teaching, in the only currency the learning loops read, that
truthful failure-reporting is bad behaviour. That is exactly the incentive gradient that
produces fabricated success, the failure mode half this project's guards exist to catch.

**What did NOT change (the guardrails):** REFUTED still beats everything (priority 1); a
shape-heuristic FAILED (abort marker, 4× selector thrash, 3× identical tool error, aborted
browser sequence) is still never upgraded — that failure is BEHAVIOURAL, not environmental;
an execution failure with NO verdict still lands FAILED; and the verifier's own priority-1
request-alignment check still refutes replies that don't answer the ask, so an off-topic
non-answer can't ride the rule to PASSED. Net: "the tool broke and I said so" = PASSED;
"I thrashed and gave up" = FAILED.

**Both delivery paths (the subtle half).** Sync folds the verdict in at write time → PASSED.
ASYNC (production) writes FIRST, so such a turn lands FAILED/"structural failure" and the late
CONFIRMED was blocked forever by the PASSED direction guard — the two paths disagreed on
identical evidence. `_backfill_trajectory_outcome` now upgrades a FAILED whose reason is
exactly `STRUCTURAL_FAILURE_REASON` (shared constant, writer+reader, so a key≠serializer drift
can't silently disable the rule — the recurring defect class) and clears the stale reason.
Refuted / shape-heuristic FAILED stay un-upgradable.

**Calibration needed NO change** — `grade_turn_outcome` already checks the verifier arms before
the execution-failure penalty (verified while implementing). The operator's Turn Outcome line
now mirrors the same priority and says "N tool failure(s), honestly reported" instead of
claiming a recovery that never happened.

Tests: test_outcome_consolidation.py (+9: priority matrix incl. refute-still-wins and
shape-not-upgraded, both delivery paths, reason-string contract, upgrade/no-upgrade pair that
is mutually discriminating). Two of my own earlier pins updated (the eval self-checks in
eval/tasks.py all still hold unchanged — verified). Docs: docs/core/agent.html new
"Honest-failure rule" section, docs/algorithms/skill_acquisition.html direction-guard
exception.

**LIVE-VALIDATED post-restart (suite 10144 green, listener 46936), verified against the CORPUS
not the log:** the "NOPE" turn (req F0/85a2d9ce) wrote `outcome: failed, failure_reason:
'structural failure'` and the late CONFIRMED then appended `{"outcome": "passed", "source":
"verifier_late"}` to the corrections sidecar — the exact upgrade the old direction guard made
impossible. Full battery re-run clean: evaluate (exact coords), verify_ui (verdict JSON),
browser+vision in ONE response (batch-order barrier fired, verified·0.90), visual gate (VISUAL
CONFIRMED 100%, verified·0.90), fail→recover (verified·0.89). Repair-fire counter 18→18 across
the battery — the corruption fix continues to hold.

**FOLLOW-UP SHIPPED (operator asked): the Turn Outcome line now SELF-CORRECTS.** In
async-critic mode the line is printed before any verdict exists, so an honest-failure turn
flashed `failed` (and a refuted turn flashed `ok`) while only the corpus got fixed.
`_record_late_verdict` → new `_emit_late_outcome_correction` re-renders the line from a
bounded 32-entry snapshot ring captured at finalize:
`turn outcome  CORRECTED failed → verified (late verdict) · confidence 0.93 · tools:
file_system · 4 chars · 1 tool failure(s), honestly reported`.
Two anti-noise properties, one of them found BY a failing test rather than by design:
(1) **valence gate** — emits only when the label crosses the failure boundary; `ok → verified`
is more information but NOT a mislabel, and announcing it would have put a correction line on
EVERY successful async turn (i.e. the first draft was noise-generating; the test that asserted
"agreeing verdict stays silent" caught it); (2) **once per turn** — the snapshot is popped on
emit so an escalation backfill can't re-announce. The priority ladder now lives in ONE shared
helper `_turn_outcome_label`, used by the finalize line AND the correction — two copies would
have drifted on the first edit, which is the exact defect shape this whole cluster kept
surfacing. Also noted: pretty_log truncates the tail in the stream, so the state change leads
the message (the suffix is supplementary). +7 tests in test_outcome_consolidation.py.

**Watch:** PASSED-with-tools volume in the corpus (skills-auto graduation input) — this rule
admits a class that was previously excluded; if graduation starts minting skills whose
"success" was an honest error report, tighten by requiring the verdict AND a non-empty
successful tool call rather than reverting the rule.

### 2026-07-31 (later 2) — Native tool_call corruption ROOT-CAUSED: the agent was teaching a second dialect

**Headline: the "merged multi-tool reply" corruption (guarded since 2026-07-05, memory says
"suspect the native path") is NOT llama.cpp flakiness — it is the agent's own legacy XML tool
prompt riding the NATIVE path. Ablation-proven, deterministic, fixed by a header split,
validated 13/13 clean.**

**Frequency finding:** 18 repair fires in one 2.5h log window (110 requests / 599 turns) —
self-play and interactive alike. Single calls usually survive the parser, so it presented as
"occasional browser/file_system weirdness"; in reality effectively EVERY stacked-call reply
corrupted (browser+vision, read+read — the pairs the model batches most).

**Ablation against the live upstream (llama.cpp b10180, template froggeric-v21.3, 31 trials):**
two-call demands parse CLEAN with simple schemas (5/5), real registry schemas (5/5), streamed
with proper per-call index deltas, and with an attribute-dialect example planted in history
(3/3). They corrupt 8/8 DETERMINISTICALLY the moment the full QWEN_TOOL_PROMPT is in the
system slot — which the native branch spliced (agent.py ~12890: schemas swapped for a pointer,
every XML format rule kept). Taught two dialects (agent `<function name="…">` attribute-style
× template `<function=…>` equals-style), the model emits a HYBRID (captured: `<parameter=path>`
mixed with attribute closers) and the server's incremental parser swallows everything after
the first parameter into that argument value → one merged call, the exact repair signature,
leaked `</parameter> </function>` fragments included. Removing the XML block: 13/13 clean.

**Fix (deployed with tests+docs):** new `QWEN_TOOL_PROMPT_NATIVE` (think-discipline + parallel
invitation, ZERO XML format instruction — the template owns the format) spliced on the native
branch; legacy branch keeps `QWEN_TOOL_PROMPT` (its XML rules ARE that path's contract);
SYSTEM_PROMPT's CRITICAL INSTRUCTION made path-aware (was: unconditional "MUST use … XML
tags"). The XML parser + repair guard stay as fallbacks for stragglers. Two old tests pinned
the broken design and were updated with the reversal rationale
(test_native_tools_keeps_format_scaffolding → …drops_xml_format_scaffolding;
test_system_prompt_json_tools_constraint). New: tests/test_native_tool_header.py (dialect-
marker absence, load-bearing parts kept, splice pins, path-neutral instruction).
Docs: docs/core/prompts.html new section. Repro harness: scratchpad corruption_repro*.py
(session-local; the journal narrative preserves the method).

**LIVE-VALIDATED post-deploy (suite 10136 green, listener 45779, 6-probe battery):**
- P1 parallel file_system×2 (the 100%-corrupting shape): **zero repair fires**, both calls
  executed, ok·0.89.
- P6 browser screenshot + verify_ui demanded in ONE response (the original d02db9d6 incident
  shape): **zero repair fires**, the two calls arrived as clean native calls, the
  **batch-order barrier fired** ("batch order: vision_analysis shares this batch with a
  file-producing tool") and sequenced vision behind the screenshot — verdict JSON returned,
  ok·0.90. Both prior-cluster fixes proven live in one probe.
- P2 evaluate (exact ball coords), P3 verify_ui (clean verdict), P4 visual gate
  (VISUAL CONFIRMED 100%, verified·0.86), P5 fail→recover (verified·0.89 — inline verifier
  now lands on the recovered shape) — all PASS. Repair-fire counter: 18 before battery → 18
  after, across 6 probes including two explicit parallel-call demands.

**Watch:** repair-fire rate over the next days (expect ~zero; each fire now warrants a look at
the raw snapshot — it would be a genuinely novel shape); parallel-call uptake may RISE now
that stacked calls actually work — the batch-order barrier (previous entry, fix 4) matters
more.

### 2026-07-31 (later) — Probe-findings fix cluster: all 5 OPEN items from the live probes CLOSED

**Headline: the five pre-existing defects the verification-tooling probes surfaced (previous
entry, "Probe-log findings") are all fixed, tested, and documented.** In fix order:

1. **TargetClosedError launch race (browser.py):** any op whose runner error mentions
   TargetClosedError retries ONCE after a 1.5s settle — the previous call's Chromium is still
   tearing down on the shared profile dir; the profile lock serialises our subprocesses, not
   Chromium's shutdown. Other errors never retry. `tests/test_browser_target_closed_retry.py`.
2. **Image-markdown guard (agent.py):** `browser` added to the valid-image-tool whitelist —
   a screenshot's `DOWNLOAD: /api/download/...` line is a legitimate image source; the guard
   used to false-positive whenever that line scrolled past the 4-message validation window.
3. **Visual-evidence provenance (agent.py `_select_visual_evidence`):** paths appearing as an
   `out_path` in this turn's tool calls (native JSON args — parsed structurally, since
   json.dumps escapes inner quotes past any blob regex — or XML parameter form) are RENDERED
   and count as after-evidence even when the user named the filename; a "before" that turns
   out to be agent-written this turn clears to None. 3 new tests in `test_verifier_visual.py`
   (collision, XML form, genuine-before preserved).
4. **Producer→consumer batch ordering (agent.py dispatch):** when one batch holds a
   file-producing tool (browser/image_generation) AND vision_analysis, producers complete
   before vision runs (two-phase gather; tool coroutines are cold until awaited so the
   sequencing is real). Kills the repaired-merged-call race that minted a spurious FATAL
   strike. 2 new tests in `test_dispatch_pipeline_extraction.py`.
5. **Turn Outcome honesty (agent.py finalize):** the operator line now requires the failure
   to be TERMINAL (`execution_failure_count > 0 AND last_was_failure` — the trajectory-corpus
   rule) before printing "failed", and surfaces recovery as "· recovered N strike(s)".
   Verified during the fix: calibration already grades recovery-aware (grade_turn_outcome),
   the corpus wrote UNKNOWN→late-CONFIRMED upgraded it to PASSED correctly in req d02db9d6,
   and work_log's descriptive "had_failures" stays. Structural pins for the inline conditions:
   `tests/test_probe_findings_2026_07_31.py`.

Also: vision pretty_log target cap 30→72 chars (the "pinball_render_chec" truncation).
Docs: browser.html (launch-race retry), verifier.html (evidence provenance), agent.html (new
"Probe-log fix cluster" section). Note for the record: repaired-batch calls stay concurrent
EXCEPT for the vision-behind-producers barrier — full sequential execution of repaired batches
was considered and rejected (most merged batches are independent reads; the file dependency is
the only observed hazard).

**Live re-validation found the DEEPER root of the outcome mislabel (probe F2b):**
`last_was_failure` was BATCH-granular — only ever SET on failure, never cleared by a later
success in the same batch, plus a post-loop blanket `= True` in the failure-handling block —
so a fail→recover batch ("read missing file → list dir → answer") still ended True and BOTH
the trajectory corpus and the (already-fixed) Turn Outcome line branded it failed, despite
their own comments promising "the FINAL tool call actually failed". Fixed: per-result
assignment in the results loop (last processed result decides; deeper failure branches may
re-assert True) + the blanket post-loop True removed. Both orderings pinned in
`test_dispatch_pipeline_extraction.py::test_last_was_failure_is_terminal_result_granular`.
KNOWN DESIGN EDGE, deliberately unchanged: a turn whose ONLY/last tool call fails but whose
answer honestly reports it (probe F2: "read missing file → reply NOPE" → outcome failed)
still labels failed — resolve_turn_outcome priority 1 treats terminal execution failure as
ground truth over a verifier PASS; softening that hierarchy is a calibration-design decision,
not a bug fix. Probe F1 re-validated the evidence-provenance fix live: the exact
filename-collision prompt that used to skip now yields VISUAL CONFIRMED (100%).

**Deployed + live-validated (suite 10131 green, listener 43101):** F1 evidence provenance →
VISUAL CONFIRMED (100%) ✅; F2c terminal-granularity → `turn outcome ok · 0.86` WITH Strike
2/6 on the ledger (was "failed" for this shape) ✅; F3 markdown guard → browser screenshot
shown inline via its /api/download link, zero guard fires ✅. TargetClosedError retry +
producer→vision batch ordering are race-dependent (not force-triggerable) — unit-covered;
their log lines ("Browser Retry", "Batch Order") self-announce if they fire live.
OBSERVATION for the record: the upstream native tool_call corruption fired TWICE in this
probe round on parallel multi-call replies (req 36: two file_system calls merged; repaired
first call carried leaked XML as its path → garbage-path read failure). The repair guard
holds, but multi-call replies remain corruption-prone on the current llama.cpp build — the
model-side workaround (one call per response) is reliable.

### 2026-07-31 — The agent was verifying its own UI with captions: verify_ui + browser evaluate + the VISUAL gate was never alive

**Headline: request 66d64313 (pinball debugging) exposed a three-layer verification gap — the
agent's only feedback channel for "did my fix work?" was a generic `describe_picture` caption;
the exact-state probe it needed didn't exist; and the verifier's pixels-ground-truth gate had
NEVER produced a verdict on this model. All three closed, tests + docs shipped.**

**The incident (550s, 5 failed fixes).** Debugging its own pinball game, the agent needed one
fact — "did the ball exit the launcher channel?" — and verified five successive wall-geometry
edits with screenshot → `describe_picture` round-trips (~30s each, ~150s of the request). Caption-
shaped feedback made it misjudge one frame ("ball is in the play area — fix works", contradicted
a turn later), oscillate through five hypotheses, and the late verifier REFUTED the turn because
the coordinates its claim leaned on were "not found in the vision" output — a caption structurally
cannot carry them. Same request also logged "VISUAL check skipped (vision returned no verdict)".

**Diagnosis 1 — the VISUAL gate was dead-on-arrival on a thinking vision model (silent-inoperative
class).** Reproduced live against Eva (Qwen3.6-35B + mmproj, preserve_thinking template):
`verify_visual`'s payload capped `max_tokens=1024`, the model spent ALL 1024 in
`reasoning_content` (`finish_reason=length`), `content` came back EMPTY → `_parse_json("")` → `{}`
→ None → skip, every time. The current log window contains ZERO successful VISUAL verdicts. With
the codebase's established switch pair (`/no_think` + `chat_template_kwargs enable_thinking=false`)
the identical call answered in 163 tokens of clean, correct JSON (it even REFUTED the test claim
for the right reason: the frame showed the menu, not gameplay).

**Fixes (all deployed together):**
1. **`core/verifier.py` `_call_llm_vision`** — ships the no-think switch pair (new
   `GHOST_VISUAL_NO_THINK`, default ON) + token cap raised via `_VISUAL_MAX_TOKENS`
   (`GHOST_VISUAL_MAX_TOKENS`, default 2048) as the belt for backends that ignore the kwargs.
   The VISUAL pixels-override gate is now actually live for the first time on this model.
2. **`tools/vision.py` `action='verify_ui'`** — the question channel, agent-facing: mandatory
   `prompt` (fails fast with an instructive error), fixed JSON verdict `{answer, confidence,
   evidence, details}` judged by a UI-auditor system prompt with an UNCERTAIN-don't-guess escape
   hatch for menu/blank frames; same no-think knob; distinct result banner ("UI VERIFICATION
   RESULT (judged from pixels only)") so the verifier can tell verdicts from captions. NOTE:
   `describe_picture` already accepted a `prompt` param — the agent never used it because nothing
   taught it to; registry description + system prompt now steer UI checks to verify_ui explicitly.
3. **`tools/browser.py` interact `evaluate` sub-action** — the probe the pinball session actually
   needed: run a JS expression in the page, get its JSON value back (aliases expression/script
   healed; output capped like extract_text with true-length + truncated flag; `default=str`
   salvage for non-serialisable values; VALUE surfaced in full by the formatter). `page.evaluate`
   existed inside the runner all along (powering extract_text) but was never exposed. JS-initiated
   fetches still pass the context's SSRF route guard — capability added, no new network exposure.
   Prompt + registry now teach the split: vision for how things LOOK, evaluate for what state IS
   (sample twice around a sleep to see motion).

**Fresh-eyes review round (read-only agent on my own edits — the 2026-07-27 convention paid off
again, 5 actionable findings past a 10111-green suite):** (a) `evaluate` was the only UNBOUNDED
sub-action — `page.evaluate` is not governed by set_default_timeout, so an unsettled Promise
("await the gameover event") hung until the subprocess kill, which discards the [BROWSER_OK]
payload and with it every EARLIER action's result → now wrapped in `asyncio.wait_for` (step
timeout_ms, 1s floor) failing per-action with a "poll with sleep+evaluate" steer; (b) 4 new tests
hard-asserted env-controlled defaults, so exercising the documented GHOST_VISUAL_NO_THINK=0 kill
turned green→red → tests now pin module attrs via monkeypatch + flag-OFF branches covered +
verified green under the kill-switch env; (c) the prompts.py verify_ui example omitted the
REQUIRED `target` param (models mirror examples over schemas) → fixed; (d) `action` was never
normalized ("Verify-UI" silently fell to the generic else and returned a caption under the
wrong banner) → normalized like the prompt aliases, tested; (e) SSRF pre-flight checked only
action=="goto" while the sanitiser heals ("goto","navigate") → guard widened to match (future-
alias asymmetry, not a live hole — the runner ignores url on evaluate and the ctx.route guard
covers JS-initiated fetches); plus GHOST_VISUAL_MAX_TOKENS="0" truthy-string edge guarded, and
one test docstring corrected (a set exercises default=str, not the except fallback — a genuine
circular-ref test now covers that path).

**Tests:** `test_browser_interact_evaluate.py` (12), `test_vision_verify_ui.py` (10), +5 no-think
regressions in `test_verifier_visual.py` (incl. empty-content → None, the live shape; flag-off).
Docs: `docs/tools/browser.html`, `docs/tools/vision.html`, `docs/core/verifier.html`.

**LIVE-VALIDATED same day (4 probes on :8000, post-deploy):**
- **evaluate** (req e63f0371): one bounded interact call → `{"title":"Pinball","ballX":370,
  "ballY":560}` returned verbatim, 42s turn, verified 0.96. The question the incident burned
  5 vision calls on is now one exact probe.
- **verify_ui** (req 884726e5): clean verdict JSON (answer/confidence/evidence/details, with
  genuinely specific details — bumper point values, flipper labels). Vision round-trip **7.1s
  vs 18–31s** for describe_picture in the same log — the no-think switches measured live.
- **VISUAL gate** (req d02db9d6): **`VISUAL CONFIRMED (100%)` — the first live visual verdict
  this gate has EVER produced** (previous log windows: skip lines only).

**Probe-log findings (deficiency sweep of the 4 requests):**
1. **FIXED same hour** — both screenshot-verify turns ran under the SPECIALIST persona ("Ghost
   Specialist Activated") which had NO verify_ui steer (I'd only added it to SYSTEM_PROMPT), so
   the agent kept reaching for bare describe_picture. Steer added to SPECIALIST_SYSTEM_PROMPT
   (UI/APP VERIFICATION bullet), test pins BOTH personas, redeployed (listener 37941→39300).
2. OPEN — visual-evidence selection misclassifies an agent-rendered screenshot as "the user's
   own image" when the USER's message names the filename (probe 68033190: agent DID screenshot
   functest_visual.png but the gate skipped "no rendered after-image" because the name appeared
   in the prompt). Prefer tool-call provenance (out_path written this turn) over name matching.
3. OPEN — after the native tool_call repair splits a merged multi-tool reply (req d02db9d6),
   the split calls ran CONCURRENTLY: vision_analysis raced the browser screenshot it depended
   on → file-not-found → spurious fatal Strike 1/6 + a retry turn. Repaired calls should run
   sequentially (or file-target tools should wait on same-turn writers).
4. OPEN — the image-markdown guard ("Caught image markdown without tool call") false-positived
   on a legitimate browser-screenshot DOWNLOAD link; the agent burned a turn arguing with it.
   Whitelist browser screenshot results as image-tag sources.
5. OPEN/WATCH — req d02db9d6's turn outcome printed **failed · 0.86** (strike-driven) although
   the reply was correct and BOTH late verdicts (VISUAL + text) CONFIRMED 100% — the backfill
   corrected the corpus, but the outcome label mismatch is the anti-correlated-calibration
   shape from the 2026-07-29 night-log audit. Verify the trajectory record post-backfill.
6. OPEN (minor) — transient `TargetClosedError` on the first atomic screenshot after prior
   browser activity (launch race on the shared profile; self-healed at the cost of 2 turns).
   A single tool-layer retry on TargetClosedError would absorb it. Also cosmetic: pretty_log
   truncates vision targets at 30 chars ("pinball_render_chec").

**RETRY round (operator-restarted server, 3 probes + log sweep):** evaluate PASS (35.6s,
verified 0.96), verify_ui PASS (37.7s, verified 0.91), and the VISUAL gate produced its
SECOND live verdict (`VISUAL CONFIRMED (100%)`, req 2c5ec4b5) — the gate is now reproducibly
alive. But the uptake probe STILL chose bare describe_picture despite the persona steer being
live. ⚠️ **My explanation at the time — "the memory bus hydrates auto-learned lessons from the
incident session which embed the OLD caption workflow (freq≥7)" — was WRONG and was corrected
by a store audit the same evening; see "(later 5) — the caption-lesson hypothesis was false".
It was inferred from a hydration log line plus the observed behaviour, never checked against
the playbook.** Two follow-ups shipped anyway, and both stand on their own merits (STAGED —
need next restart; a live user request was on the box so no kill):
- **Moment-of-use steer:** a bare describe_picture (no prompt) result now carries a TIP line
  pointing at verify_ui — same pattern as browser's PRE_INTERACTION; lands mid-decision where
  prompt guidance loses.
- **Empty-vision fail-open closed:** req 2c5ec4b5 turn 2 got "" back from the contended node
  UNDER THE SUCCESS BANNER and burned 57s noticing; empty content now ships as a named
  "Vision API Error: … EMPTY result" (contract change: test_content_null_does_not_error
  updated — crash-protection half stands, silent-empty-success half deliberately reversed).

**FINAL round (post-TIP restart, listener 40061): UNPROMPTED UPTAKE CONFIRMED on the incident
shape.** Asked the exact question class that caused the original 550s failure ("is the ball in
the launcher channel or the play area, at what coordinates?" — NO tool named), the agent's
thinking chose it explicitly ("use browser's evaluate action to read the game's JavaScript
state directly"): one evaluate call → precise answer (channel, x=370/y=560, cross-checked
against the channel geometry from the source) with ZERO vision calls, verified flow, C=0.92,
97s. The VISUAL gate also produced its THIRD consecutive live verdict (CONFIRMED 100%) across
two restarts. Residual: on BROAD "does it render correctly" questions the agent still opens
with bare describe_picture — defensible there (a caption answers a broad question), and that
result now carries the verify_ui TIP for the moment it isn't.

**Watch:** ~~whether the stale caption-workflow lessons decay or need a manual scrub~~ —
RESOLVED by audit, there are no such lessons (see "(later 5)"). Kill switches:
`GHOST_VISUAL_NO_THINK=0` (restores thinking for BOTH visual paths), `GHOST_VISUAL_MAX_TOKENS`.

### 2026-07-30 (later 3) — Auth orphaned sandbox apps → service tokens; a false premise looped the solver

**Headline: the 2026-07-13 auth rollout silently broke every sandbox-hosted app that integrates
with the agent — discovered via three "auth rejected" log lines, fixed with supervisor-minted
service tokens.** The released Chess Coach's app.py never sends X-Ghost-Key: its /api/health
probe 403'd (surfacing as `ua=Python-urllib/3.11` rejections at every service start), base-URL
resolution then fell through to a WRONG hardcoded 127.0.0.1:8000, and every coaching call would
403 — players get "Could not reach Ghost". The release rehearsal (TCP probe) structurally cannot
see an app→agent auth failure. Also identified: `[own functional suite]` rejections = the
deliberate self-test probes (working as designed), and the `/v1/images/generations` rejection =
`interface/externals/image_generation/test_gen.py` with a hardcoded loopback JETSON_IP (patched:
env-overridable, defaults to the Jetson's tailnet IP).

**Operator-instruction postmortem (request 1d588fd5, FAILED):** the fix instruction to the agent
claimed /api/game/* was "unauthenticated by design" — FALSE (`game_router` declares
`Security(verify_api_key)`). The solver correctly observed 403s contradicting the instruction and
spiraled: repeated-paragraph loop at 10k chars (aborted), then thinking cap 2x → attempt aborted.
Guards all worked (paragraph detector, cap, native-corruption `content==replace_with` catch, edit
churn steer, SSRF block on browsing the agent API); the §4G machinery ALSO worked (v2 fork got
`chess-coach-v2` project-scoped on auto-granted port 8101 next to v1's 8100). Lesson: a false
fact in an instruction is a LOOP SEED — the model can neither comply nor refute its operator.

**Fix — service tokens (sandbox/services.py + api/game_routes.py):** supervisor mints
`$GHOST_SERVICE_TOKEN` per start (32 hex, stored on the registry entry, exported in cmd.sh);
`verify_game_access` on the game router accepts master key OR a live token via
`X-Ghost-Service-Token` — scope is GAME routes only (token never unlocks /api/chat — test-pinned),
registry-driven revocation (stop/stop-all; restart mints fresh), constant-time compare, fail-closed,
briefing never surfaces tokens. The /api/game/move response already carries move + comment +
move_explanation + critique, so the chess app's coaching maps onto the participant endpoint with
no /api/chat need. Chess Coach v2 (ac67f4418187, service on 8101) awaits the app-side rework +
release AFTER this deploys. Known pre-existing test-order artifact: test_sandbox_services'
asyncio.run closes the main-thread loop → test_auth_rejection_logging fails if run AFTER it in a
custom order (canonical alphabetical order unaffected).

### 2026-07-30 (later 5) — Released workspaces are read-only, but service apps write runtime state into them → $GHOST_SERVICE_STATE_DIR (supervisor half SHIPPED)

Chess Coach v2 RELEASED (20:12, dossier correct: chess-coach-v2 · 8101 · URL + directions) and the
full round-trip verified pre-release (e2e4 → e5 + coaching, twice). POST-release the app 500s on
every move: release chmod's the workspace read-only (immutability by design) while app.py writes
`game_state.json`/`saves/` INTO its workspace at runtime → PermissionError. v1 has the same latent
bug. Design collision, not a regression. Recommended fix (§4G-consistent): supervisor exports a
writable per-service state dir (e.g. `$GHOST_SERVICE_STATE_DIR` → `.services/state/<stem>/`,
created at start, survives releases); released apps keep their artifact immutable and their state
outside it. SHIPPED same evening (supervisor half): state dir created pre-launch + exported in
cmd.sh; kept across stop/restart; purged only via project hard delete
(`_stop_project_services(purge_state=True)`); tool description teaches the contract; tests in
test_service_port_leases.py. CLOSED same night: Chess Coach v3 (48e0373aaab3) released on 8102
with STATE_FILE/SAVES_DIR routed through the state dir; post-release round-trip verified TWICE
(ghost's e2e4→e7e5 + coaching; operator-side d2d4→g8f6 with the state file mtime advancing under
.services/state/ while the workspace sat read-only). ALSO: turn wrap-up gate (`force_final_response` after a task close)
dropped the follow-up release call three times today — logged as friction, not yet changed.

### 2026-07-30 (later 4) — Token race: the credential landed in the registry AFTER the app's first probe

**Live verification of the v2 round-trip FAILED and exposed a race in the service-token feature:**
start() saved the registry entry (with the minted token) only after launch + liveness + up-to-6s
port probes — but a service's FIRST act can be probing the agent with $GHOST_SERVICE_TOKEN, and
validation reads that registry. chess-coach-v2 resolves its base URL ONCE at import: its probe
fired inside the ~2-8s window, got 403, and the app latched the wrong loopback fallback forever
("Connection refused" on every move, looked like a connectivity problem). Diagnosis chain that
nailed it: app-env token == registry token ✓, in-container curl host.docker.internal with that
token → 200 ✓, probe code correct ✓ — only the TIMING was left. Fix: the entry (with credential)
is persisted BEFORE the launch exec (pid=None provisional); every launch-failure return pops it.
Regression tests capture the registry AS SEEN AT LAUNCH (`test_token_is_in_registry_BEFORE_launch`)
+ failed-launch cleanup. Lesson for the class: **when a supervisor mints credentials consumed by
the supervised process, the credential must be valid before the process can possibly present it**
— same shape as the pre-flight-guard deadlock (state written after the reader needed it).

### 2026-07-30 (later 2) — §4G SHIPPED: services are project-owned, ports are leases

**Headline: the supervisor is now the port allocator; the model never picks a port again.** All
three §4G phases implemented, review-hardened, and test-pinned in one session (follow-on from the
solar-sim postmortem earlier today). Files: sandbox/services.py (bulk), tools/sandbox_services.py,
tools/registry.py, tools/projects.py, core/prompts.py, core/agent.py, main.py, tools/execute.py.

- **Phase 1 — ownership + leases.** Registry keys scoped `<project>:<name>` (files
  `<project>--<name>.*`), `project_id` stamped from the bound project (registry lambda),
  EVERY start granted a lease (services expose output over HTTP by operator definition;
  `port=0` = persisted portless opt-out). Requested port (arg or command literal via
  `extract_command_port`) is a PREFERENCE: bind-probe (0.0.0.0, catches unregistered holders —
  the class the old alive-claim check missed), fallback through the published range with
  who-was-in-the-way notes, literal substitution when the lease moves, dead entries' ports
  deferred as their restart contract, holders never killed. Default workdir = project workspace.
- **Phase 2 — awareness.** `reconcile()` (registry ⋈ `ss -ltnp`; READ-ONLY — no auto-restart on
  agent boot, by operator requirement), one-line boot summary in main.py lifespan, `adopt` action
  registers existing unregistered listeners (cmdline/cwd/project-hint captured), `status` gains
  port map + orphans block, ACTIVE-project briefings get a SERVICES line (registry facts only,
  zero per-turn execs; staleness = container-generation mismatch), release rehearsal reads
  fresh port+command after restart.
- **Phase 3 — orphan source closed.** execute refuses detach+serverish commands
  (`_daemonized_server_block`) with a manage_services steer — the exact autoadvance
  `python3 server.py &` leak that caused the incident; effective service cap = published-range
  size.

**Fresh-eyes review (6 confirmed defects, all fixed + regression-pinned):** portless opt-out not
persisted (restart force-granted a lease, hard-failed on full range); plain-exact-first resolution
let a bound project stop a legacy 'web' it didn't own (scope now precedes); daemon guard scanned
heredoc bodies/quoted strings (now stripped, `sh -c` payloads still scanned, `kill %1` smoke-tests
pass); release dossier stored pre-restart command with post-restart port + ignored "Error:" restart
returns; explicit request over a dead reservation was silent and the port map named the dead entry
as owner; ambiguity errors said "no service named" on restart/status/logs. Plus: live legacy entry
no longer blocks a bound project's same-name start (dead → re-keyed adopt, live → scoped twin);
reserved-port command literals rewritten. Tests: `test_service_port_leases.py` (61) + updated
`test_sandbox_services*.py`. Requirements verdict from review: no-auto-restart HOLDS, occupied→
different-port HOLDS, multi-project HOLDS, never-kill-holders HOLDS. Known watch items: RELEASED
dossiers can go stale if a post-release restart moves a port (rehearsal now notes moves at release
time); lock held across container execs in start/restart (pre-existing pattern) serializes
mutating service ops if docker wedges. NOT yet deployed at write time.

### 2026-07-30 (later) — Pre-flight guard deadlock: the block that outlived its own remediation

**Headline: the repeat-failure guard blocked a VERIFIABLY-FIXED `manage_services start` across
three consecutive requests (A3 16:51 → 92 16:57 → E7 17:02), each request dying at the 2-block
budget.** Arc: autoadvance left an unregistered `python3 server.py` on :8102; `start solar-sim`
failed twice ("exited immediately", port taken) → guard armed; the model then did everything right
(found PID 636 via `ss`, killed it, confirmed the port free) — and every subsequent start was
blocked: same port, different port (:8103), different command. Four root causes in
`RecentFailureGuard` (core/triggers.py), all fixed + tested + docs:

- **Self-deadlock — `reset()` had ZERO callers.** A blocked call never dispatches, so it can never
  refute stale entries; the 2026-07-18 world-changed fix only reached StrikeLedger. Now
  `note_world_changed()` clears the guard on any SUCCESSFUL state-mutating call: file_system
  mutation, mutating manage_services action, or an `execute` command matching the new
  `looks_mutating_command` heuristic (kills/`fuser -k`/file verbs/service-manager subcommands/bare
  redirects; NOT `kill -0`/`pkill -0`/fd-redirect probes; verbs must sit at command position and
  quotes are stripped — except `sh -c '…'` payloads, scanned as commands). Deliberately global +
  permissive: false clear = one extra real attempt (threshold re-arms), false block = this bug.
- **Key collapse.** `primary_target_from_args` knows none of manage_services' identity args
  (name/port/command) → every start of every service shared `(manage_services, "", start)`.
  New `guard_key_target(primary, a_hash)` falls back to an `args#`-SHA1 of the full canonical call
  → only byte-identical re-issues match; check site and record site share the helper.
- **Stale cross-request memory.** Guard lives on the agent instance; entries only aged out via NEW
  failures. Now reset at request start (next to `strikes = StrikeLedger()`); cross-request
  pathology stays the offline post-mortem's job.
- **(Review catch) `execute` failures wear an `EXIT CODE:` banner, not `Error:`** — so a FAILED
  remediation (`kill` on a stale pid, exit 1) would have counted as a successful mutation and
  CLEARED the guard. The world-changed branch now gates on `_pf_exec_failed` (mirrors the execute
  strike branch's exit-code parse). Deliberately asymmetric: execute failures still are NOT
  recorded into the guard — first attempt at symmetry broke `test_system3_crisis_pivot` (identical
  failing shell re-issues are the strike ledger's / System-3 pivot's crisis signal; pre-dispatch
  blocking starves the pivot). The fresh-eyes review agent also killed regex FPs on
  `grep -rn 'mkdir'`, `ls *.tar.gz`, `awk '$3 > 100'`, `python3 -c 'print(1 > 0)'` — probe
  traffic dominates, so FP-frequency = guard permanently dark.

Observability: a clearing reset logs "World changed (successful <tool> mutation) — cleared N
recorded failure(s)". Tests: `test_preflight_guard.py` (world-changed/signature/heuristic units +
wiring introspection), `test_guard_box_fixes.py` updated. Docs: `docs/core/agent.html` (full
postmortem section), `docs/algorithms/metacognition.html` (lifecycle paragraph). NOT yet deployed —
live agent still runs the old code until restart.

### 2026-07-30 — §4F overnight: verifier prompts SHIPPED+LIVE (+0.087 private), Phase 3 built, B4 staged

**Headline: the first GEPA-optimized prompts are serving production traffic.** Verifier two-stage
templates optimized against the REAL pipeline (custom gepa adapter over verify_bench fault-injected
trials, judge = the live worker endpoint), gate-judged on 23 PRIVATE trials the optimizer never saw:
**baseline 0.796 → candidate 0.883 (+0.087 ≫ 0.02 gate) → PROMOTED → agent restarted → both
templates confirmed LOADED on a live tool turn** ("GEPA: loaded tuned instruction for
'verifier.enumerate' (2035 chars) / 'verifier.adjudicate' (5366 chars)") — unlike Phase 1's dark
planner, this read-site is hot on every substantive turn. Pre-ship instrument reading (Gemma judge,
two-stage): TPR 0.80 overall but **FPR 0.31** and degraded-evidence FP 0.78 — exactly the
false-positive surface the adjudicate prompt owns. Post-ship bench + Phase 3a probe A/B: see the
appended reading below.

**Phase 2 verifier machinery built this session** (`scripts/optimize_verifier.py` + verifier.py
changes): tunable stage templates behind `_stage_template` (override → loader artifact → baseline)
with a **probe-format placeholder guard** (a candidate that lost `{claim}` or broke `{{ }}`-escaping
falls back to baseline instead of raising in verify_claim); custom gepa `GEPAAdapter` whose
`evaluate()` runs REAL `verify_claim` over bench trials (fresh HttpChatClient per event loop —
httpx pools are loop-affine) and whose reflective dataset names the injected fault each wrong
verdict missed; bench CASES hash-split public/private (`holdout_tier("vbcase:<id>")`);
`--run-dir` gepa checkpointing after a session-transition kill vaporized a 90%-complete run
(watchers die with sessions — the optimizer itself now runs nohup-detached). Bugs found live:
gepa probes the OPTIONAL `propose_new_texts` adapter attr by direct access (must exist as None —
first fixed run completed with zero mutations and correctly self-rejected at the gate);
model-churn note: nova briefly served Agents-A1-4B (confirmed confirming corrupted claims at
conf 1.0) before operator reverted to Gemma — the bench measures the JUDGE, always re-baseline
after judge swaps.

**Phase 3 (§4F) built, default-OFF per §3 doctrine (no unproven layer live without a measured
win):** (a) logit-expectation confidence probe (`GHOST_VERIFY_LOGIT_EXPECT`): one bounded digit-
scale score call after two-stage verdicts, expectation over top-logprobs digit mass →
`VerifyResult.probe_score`, blended 50/50 verdict-aligned into confidence; verdicts never change;
probe failure leaves the result untouched; always rides the cheap pool. (b) wobble-band adaptive
best-of-N (`core/tts.py` + `_adaptive_bon_final` hook at the loop-exit verifier gate;
`GHOST_TTS_ADAPTIVE_BON`, `GHOST_TTS_BON_K`): fires ONLY on UNCERTAIN or sub-0.7 REFUTED (hard
REFUTED keeps auto-repair — mechanisms never interact); K sequential diversified candidates → ONE
list-wise comparative judge call → winner substitutes; every failure resolves to the original.
Judge payload lives in tts.judge_payload — agent.py stays under the one-disable-thinking-switch
guard (test_self_play_redesign caught the migration; moved, guard re-armed + a tts-side twin).
(c) verified-restart: assessed as substantially PRE-EXISTING (auto-repair = critique-conditioned
restart w/ narration discard + round caps); summary-conditioning delta deferred as marginal.
Tests: +12 probe, +23 tts; suite 9917 passed (2 FORCE_COLOR env artifacts pass in clean shell).

**B4 battery staged for the §4F acceptance rule** (operator asked mid-wait): infra from 2026-07-09
verified alive (103 tests, zero drift), candidate pool grew 22→35 across 7 clusters + held-out
web_automation; pilot emitter now ALSO writes `b4_battery_tiers.json` (hash-stable public/private
via `holdout_tier("b4:<id>")` — full-pool preview 28/7). Pilot logging made LIVE (per-task
PASS/FAIL + incremental records + per-pass aggregate) after the operator flagged the B3-ceiling
risk pre-commit.

**B4 PILOT RUN + ABORTED (2026-07-30 ~14:05): THE POOL IS SATURATED — operator's call, confirmed
in 41 min instead of 2.5 h.** Pass 1 on a control-configured agent: **33/33 PASS, zero failures**
(run killed at 33/35 per the pre-registered >80% abort rule; partial records in
`ablation_out/b4-pilot-20260730/b4_pilot_records.partial.json`). Reading: the 07-09 pool was
designed to produce real failures and the CURRENT agent ceilings it — a genuine capability datum
(three weeks of self-play on exactly these cluster families + the July fix cohorts; near-ring
isomorphism to challenge_templates makes contamination-by-training the default suspect, §4D's
own contamination guard notwithstanding). Twist-stacked messy data does NOT defeat a
write-run-verify sandbox loop at this scale. HARDENING FORK (operator decision pending):
(a) twist/scale escalation — mechanical ~1 h, HIGH risk of a second ceiling; (b) COMPOSITIONAL
tasks (chain existing shapes, A's artifact feeds B; compositional depth is the measured 2026
difficulty axis) — ~half-day of honest verifier plumbing, recommended core; (c) new far-ring
families self-play never trained (only 2 web_automation shapes exist) — most authoring, best
contamination profile. Battery remains the Phase-4 gate; §4F acceptance rule unchanged.

**Hardening option (b) DELIVERED same day (operator-picked): 10 compositional tasks**
(`load_b4_comp()` in trackb4_tasks.py, ring="comp" — a DEPTH axis orthogonal to transfer rings;
structure test updated for the new contract). Two-stage chains over all 7 clusters + the held-out
family (clean→rank, extract→aggregate, dup-resolve→join-total, edge-clean→BFS, rotated-merge→
windowed-count, tie-broken JSON merge→filter-sum, concurrent per-file sums→median, comment-aware
HTML extract→top-dept, pivot→largest-drop, ledger reconcile→abs-diff total). References are pure
chained functions (stage-A error propagates to the graded artifact); prompts contract intermediate
files; verify remains final-artifact token containment. 123 battery tests green (auto-swept).
Pilot economics: `--only-ring comp` added — re-pilot = 10×3 runs ≈ 30-50 min prod-down, not 2.5 h.
Comp tier preview: 8 public / 2 private (thin — if private-side survivors < ~3, raise the b4:
namespace private pct before first §4F use).

**COMP PILOT RESULT (2026-07-30 15:41): 10/10 PASS — the compositional pool ceilings too** (aborted
at pass-1 aggregate per the >80% rule; ~28 min prod-down; slowest task comp_web_table 172 s, still
solved). TWO difficulty models falsified in one day: neither messy data nor 2-stage composition
defeats this agent's write-run-verify loop on these task families. Honest reading: 2-stage chains
sit far below the literature's collapse thresholds (failures concentrate at 5-10+ DEPENDENT steps
with state, hidden information, mid-task changes, and constraint retention across long horizons —
not at depth 2), and three weeks of self-play on these exact cluster families moved the envelope.
DESIGN FORK for the instrument (operator decision): (a) deep discovery-chains — 5-8 stages where
each stage's SPEC is revealed only by the previous stage's output (defeats one-shot scripting,
stresses multi-turn coherence; same pure-reference pattern, needs per-task timeout > 300 s);
(b) long-horizon behavioral tasks matching where prod actually fails (autonomous project work, web
research under uncertainty, constraint retention) — highest validity, 10-30 min/task makes piloting
expensive; (c) accept the ceiling as a FINDING: the agent has outgrown sandbox data-task batteries;
measure §4F phases on live-trajectory outcomes instead (observational, the earn-keep post-mortem's
"instrument changes" route). Costs sunk so far are small (2 aborted pilots ≈ 70 min downtime) and
the abort instrumentation works.

**Post-ship instrument reading (full 13-case bench, Gemma judge, two-stage):**
- **Tuned templates: TPR 0.80 → 0.893** (artifact_leak 0.385→0.846, fact_swap 0.50→0.90 — the two
  worst classes fixed); **FPR 0.31 → 0.385 (WORSE)**, degraded-evidence FP 0.78→0.889. Honest read:
  the optimizer traded false negatives for false positives, and the gate's composite rewarded it
  because REFUTED-expecting trials outnumber clean ~5:1 in the trial mix — a metric-weighting
  artifact to fix before the next optimization round (weight clean/NOT_REFUTED trials up).
  KEEPING the ship: it passed the pre-registered gate, and in prod the refute-escalation bounds FP
  damage (cheap-judge REFUTED must be confirmed by the main model before acting — false refutes
  cost latency, missed catches cost correctness). WATCH: live false-refute churn (correction dedup
  + `escalated_overturn` counts) over the next log audits.
- **Phase 3a probe A/B (tuned + GHOST_VERIFY_LOGIT_EXPECT=1): default stays OFF.** Raw rates
  similar (TPR 0.92 / FPR 0.462 — judge nondeterminism), but the 50/50 blend drags nearly ALL
  confidences below the 0.7 actionable gate: actionable FPR 0.0 (good) at actionable TPR 0.347
  (unacceptable — would neuter the verifier). Signal exists but the blend is miscalibrated;
  revisit with a lighter blend weight or threshold-aware calibration, never a straight flip.

### 2026-07-29 (later 7) — §4F Phase 1 IGNITED: first GEPA candidate PROMOTED through the private gate

The never-run GEPA loop ran end-to-end for the first time. **Result: `planning.decompose` candidate
promoted — public valset 0.401→0.589; A/B on the PRIVATE 20-example holdout: baseline 0.45 →
candidate 0.80 (+0.35 ≫ 0.02 gate)** → live artifact `$GHOST_HOME/system/optim/planning.decompose.json`.
The evolved instruction (self-reflected by the 35B from its own trajectories) mandates a strict
`### plan`/`### rationale` structure, a 4-step logical flow, explicit empty-tools handling, and a
constraint-encoding template — independently converging on the "constraint loss dominates
long-horizon failure" finding from the 2026 literature. **Deploy = restart** (loader caches
per-process); success check = learning-health `PROMPT OPTIMIZATION` activation counter > 0 in prod.

**Ignition flushed 6 written-but-never-exercised defects** (the §4F prediction, on schedule):
1. Trajectory-root drift: script + `collector._default_root()` pointed at `$GHOST_HOME/trajectories`;
   prod writes `$GHOST_HOME/system/trajectories` (main.py memory_dir.parent). Both fixed + regression
   test pinning default-root to the prod path.
2. Raw `TrainExample`s fed to dspy (needs `dspy.Example` bound to signature FIELD NAMES) →
   `_to_dspy_examples()` with ""-defaults for missing inputs; drops expected keys not on the signature.
3. Metric arity: dspy 3.x GEPA isinstance-checks a 5-positional metric `(gold, pred, trace,
   pred_name, pred_trace)`; the shipped 2-arg metric died at construction.
4. Binary substring metric scored ~everything 0 (no gradient) → graded token-recall `_overlap`,
   shared verbatim by optimizer metric and A/B runner (same semantics both sides of the gate).
5. `_GhostLMAdapter` hit its THIRD dspy-interface break (BaseLM isinstance gate) → GEPA now uses
   `dspy.LM` pinned to the same local `/v1` endpoint (adapter retained for other callers). Thinking
   models need max_tokens ≥ 8k or reasoning eats the budget and content returns EMPTY
   (`chat_template_kwargs.enable_thinking=false` for rollouts; reflection keeps thinking).
6. The A/B runner had the same empty-content bug (1024 tokens, think-on): both arms scored at the
   noise floor (0.05/0.00) and the gate could only ever reject — measured live when a TaskStop-orphaned
   first attempt completed with exactly that verdict. Fixed to mirror the rollout regime; rejected
   candidates now kept as `.candidate.rejected` for post-mortem instead of deleted.
Also: field-coherence finding — the generic trainset only matches `planning.decompose`;
`tool_selection.pick`/`reflection.critique` need signature-specific example extractors (Phase 2).
Ops notes: GEPA runs ~35 min cold on the box (~11 s/rollout think-off; dspy disk cache makes reruns
minutes); kill by PID (TaskStop on the pipeline orphans the python); one log file per launch.

### 2026-07-29 (later 6) — Methodology survey → §4F 4-phase plan; Phase 0 eval hygiene SHIPPED

Four-agent web survey of mid-2026 agentic-engineering SOTA vs this stack (full findings + arXiv ids:
memory `agentic-methodology-survey-2026-07`; plan: **§4F**). Cross-validated top-3 gaps: text-space
prompt optimization (GEPA/SkillOpt), trajectory-level verifier-guided TTS, long-horizon context
discipline. **The finding that reordered everything: the GEPA loop (`optim/`) was fully built and
read-wired but had NEVER RUN** — no `$GHOST_HOME/system/optim/` exists; 24 days of trajectories
unused. Also resolved the model watch-item: Agents-A1 35B-A3B shipped open (GGUFs 07-02, 4B 07-14,
bake-off pending); open Qwen3.7 is dead (API-only pivot) — memory updated.

**Phase 0 (eval hygiene) shipped this session** — the precondition for ever running the optimizer
(proxy-gaming hits 46–74% of self-optimization runs, RISES with steps, and only a hidden holdout
reliably prevents it):
- `optim/trainset.py`: `holdout_tier` + `split_public_private` — per-item sha256 PUBLIC/PRIVATE
  split on `source_trajectory_id` (content-key fallback; identity excludes signature_name so one
  trajectory is same-tier for every signature). Membership can never migrate as the corpus grows —
  the old seeded-shuffle `split_train_eval` re-deals membership every run (slow leakage). Public
  never starved (rescue-one rule).
- `scripts/run_gepa.py`: optimizer + its internal val now see ONLY public; the A/B ship-gate judges
  ONLY private (`--private-pct`, default 30); empty private ⇒ refuse promotion. The public val split
  is passed through as `run_gepa(..., valset=...)` (GEPA selects its Pareto frontier on it —
  public-tier by construction; TypeError fallback for tuners without valset).
- `optim/run_gepa.py`: `MAX_OPT_ITERATIONS=16` clamped inside `run_gepa()` (all callers inherit).
- `optim/loader.py`: activation counters (applied vs fallback per signature, survive `clear_cache()`)
  + `activation_stats()`; docstring hardened: NEVER `clear_cache()` live (KV stable-prefix re-prime).
- `core/learning_health.py`: `optim` section pairs artifacts-on-disk with in-process counters;
  render flags "tuned but 0 applies since boot" ⚠ — the write-only defect class this loop already
  exhibited once, made visible in `introspect action='learning'`.
- Tests: `tests/test_optim_eval_hygiene.py` (15, incl. membership-stability-under-growth and
  clamp-before-optimize); optim/gepa/learning-health regressions green (51+15). Docs:
  `docs/self_improvement.md` "Optimizer eval hygiene" section. Next: Phase 1 ignition run on
  `tool_selection.pick` (needs llama-server idle window).

### 2026-07-29 (later 5) — Web UI: the 'cube' face (infinite monolith with a growing mutation)

**v2 rework** (operator: cube unclear / mutations must be red + evolving at idle / zoom too deep, "random
dots" / erratic zoom-out): cell edge 0.88→0.80 + edge/corner nodes render larger (wireframe SILHOUETTE =
legible monolith) + near-still per-form heading wander; complexities idle as slow RED organisms (S_total =
0.22+0.12·sin breath + active cubeS; hearts 0.52+ crimson at rest); displacement field made SPATIALLY
COHERENT (position-keyed low-freq phases, per-node jitter down to texture — neighbors move together =
spreading flesh, not dots); dive PARTIAL for this form (CUBE_DIVE_Z 3.1, swell 0.18, lookAt −1.0 — watch
the spread, never enter the dot cloud); erratic zoom-out fixed by pacing (taming 0.988→0.9935 ≈ dive-out),
gentler accretion (0.55→0.35) and the damped wander. Full lifecycle re-verified live, 133 interface tests
green (lookAt pin made form-aware).

Operator concept, distilled from what they loved in lattice ("a large dark cube with alien complexities in
some parts of it"): an infinite-reading dark grid (cell edge 0.88/0.95, spans past the viewport) with 3/2
resident complexities — localized alien sine fields + violet heart tangles — quietly deforming it. A USER
turn wakes ONE (userTurnState rising edge, vortex pattern; ambient work doesn't): eased growth ~5s
("aggressively but not overly fast"), influence ×2.6, churn speeds with strength, per-node irregular `gate`
spread boundary, links tear/re-weave, ACCRETION (0.55·S·w — two renders read as scattered stars until the
pull got wide/strong) weaves the crimson knot the dive lands on; dive focus-translated onto the active
anchor (descent/embedding lesson applied from birth). Completion: ×0.988 taming tail, cube re-knits, dive
eases out; mid-decay re-arms keep the same anchor (heat never teleports). Full lifecycle live-verified
headlessly (idle/growth/full-immersion/recovery, zero JS errors). FORMS now 10 ('cube' before 'empty');
menu auto-lists it (hint "infinite monolith"). Cache-bust app/matrix 7.5. Pins: `test_cube_mutation_contract`,
`test_cube_links_neighbors_but_not_diagonals`, node-harness budgets.

### 2026-07-29 (later 4) — Web UI: four AI face forms (lattice / stack / embedding / descent)

Operator: same nodes-lines-animation design as the 2026-07-28 forms, different shapes, AI-themed. Four
new body plans over the shared motion engine — the machine's own internals as anatomy, first ANGULAR
silhouettes: **lattice** (tumbling weight-tensor grid, diagonal activation waves, drifting hot attention
kernel, axis charge-runners), **stack** (tapering transformer layer-rings + hot residual column + climbing
token packets that ripple each layer), **embedding** (octahedral concept clusters + hot query comet doing
bezier-arc recall — arrival ignites the cluster), **descent** (evolving loss-landscape sheet, TRUE
gradient-descent bead with damping/soft-walls/stuck-kick, flinch = bad gradient step). Cycle keeps 'empty'
last. Operator picked all three structural ideas + embedding from six proposals (loom/swarm passed on).

Engineering: form dispatch refactored to NAME-based (`const FORM = FORMS[formIndex]` — the hardcoded
`formIndex === N` comparisons would have silently mis-dispatched when FORMS grew past 5); per-form link
radii centralized in `LINK_MULT` (lattice 0.45 = neighbors-only wireframe; stack 0.25 = discs must not
scaffold into a cylinder, which the first render did; embedding 0.62 = clusters can never cross-link;
descent 0.20/0.38 = mesh must survive the worst-case analytic slope — my own test caught the 0.16 margin
violation). All four reheat `nodeSeeds` per frame (aSeed upload like vortex). Verified: builders executed
under node for BOTH device classes (exact NODE_COUNT fill — mismatch corrupts instanced attributes
silently), all four forms rendered live on :8080 via Playwright (anatomy + pulse + zero JS errors,
screenshots reviewed; only stack needed the link tuning). Cache-bust 6.7→6.8 both files. Tests:
`tests/test_interface_face_forms_ai.py` (incl. computed geometry invariants + node-executed budgets);
`test_interface_face_palette.py` vortex pin updated to the LINK_MULT shape. Docs:
`docs/interfaces/web_server.html` "AI face forms". No server/agent changes — static-only, live on next
browser reload.

Follow-up 3 (operator: "any other UX improvements?" — picked the activity ticker from a 7-item survey):
**turn status line**, placement iterated three times to the operator's final spec: v1 breadcrumb trail
inside the thinking bubble (rejected), v2 caption next to the center-stage 2.2rem activity icon (rejected —
icon dwarfed it), v3 FINAL: `timer : description - icon` with a text-sized icon, re-parented directly UNDER
the waiting reply bubble each turn (parked hidden between turns; survives /clear repaints via re-adoption).
Friendly descriptions via TICKER_VERBS map + detail-column suffix ("delegating to the worker node ·
decompose query…"); think-class → "thinking… - 💭", first content chunk → "writing the reply… - 💬"; hides
at turn end. Corridor-id adoption keeps background self-play/dream lines out (first "request started" after
send = ours); TICKER_NOISE plumbing filter. All three versions live-verified with a real tool turn each
(test sessions deleted after). Then the center-stage #activity-icon was REMOVED outright (operator:
"remove the icon from the middle") — element, container, dwell/priority arbitration, flash/fade, zen-scale
rule all excised; updateStateFromIcon kept as the face's working-state driver; live-verified (icon gone,
status line intact, zero JS errors). Integration polish (operator: "the timer doesn't belong there"): the
clock became a quiet tabular-nums CHIP (violet-bordered pill — a badge, not part of the sentence; no width
jitter at 0:09→0:10), the description moved to the UI's main font, literal `:`/`-` separators replaced by
visual grouping (style 4.8). Cache-bust app 7.4 / style 4.8. Pin: `test_turn_status_line`.
Remaining survey items (not built): jump-to-latest pill, draft persistence, retry-last-turn, session
export, full-text session search, shortcuts overlay.

Follow-up 2 (operator): (a) **delete-all sessions** button at the rail bottom — two-step armed confirm
(first click arms danger-red with the count, second within 4s executes; timeout/pointer-leave disarms),
iterates the enumerated per-id DELETE proxy (no bulk endpoint added), lands on a fresh session, hidden when
the list is empty; (b) **face-form picker** replaced the blind cycle button (9 forms = up to 8 clicks ×
1.4s blends) — a glass menu built from matrix_graph's own `getForms()` roster (new forms auto-appear),
name + hint per row, current marked, `setForm()` jumps directly, stale-cache cycle fallback kept.
Verified live headlessly (menu items/active/switch; delete-all with DELETE interception — real sessions
untouched). Cache-bust: app/matrix_graph 7.0, workspace→sessions 6.8, style.css 4.3. Pins:
`test_delete_all_sessions_button`, `test_form_picker_menu`.

Follow-up (operator: "descent usually zooms into an uninteresting location when busy"): the immersion dive
targets the scene ORIGIN — for descent a generic terrain patch, for embedding the deliberately-empty void
between clusters. Both forms now translate their space while diving (dive-weighted, zero at rest) so the
dive rides the form's HOT FOCUS: the optimizer bead (+0.30 lift so the camera hovers over the surface) /
the query comet mid-recall. Both foci are positionally continuous, so the follow never jumps. Verified at
full dive via Playwright (immersion 1.0, camZ 1.3, screenshots). Cache-bust 6.8→6.9; pin
`test_dive_centers_on_the_form_focus`.

### 2026-07-29 (later 3) — Daytime-log audit: the two ungated self-play spots compounded

Read the morning's `ghost-agent.log` (boots 09:47/09:49 → 12:38) for bugs/corner cases/improvements. Headline:
the guards shipped over the last weeks all fired and did their jobs (loud selftest skip, consistency discard,
calibration rejection, tool-call repair) — today's failure concentrated in the two spots WITHOUT a gate, and
they compounded: the selftest instrumenter exit-43'd on the generated validator's shape (echo gate dark), and
the sim short-circuit then shipped a wrong-FORMAT solution to that unverified validator on exit 0 alone
(request `9D`: per-resource debug lines vs `TotalAllocatedQuantity: <sum>` → judge FAIL, attempt burned).

**Eleven fixes, all tested** (`tests/test_log_audit_fixes_2026_07_29.py`, 62 tests; full suite green):
1. **Short-circuit format gate** — `_challenge_output_prefixes` mines a pinned literal output label
   (`Label: <value>` templates only; column sketches never veto) from the challenge; short-circuit requires it
   in stdout, else the confirmation turn is KEPT so the model can fix the print format.
2. **Selftest instrumenter widened** — innermost-statement insertion (incl. inside `def main():`, indent-aware,
   file kept not truncated), loosely-named holders (`exp_out`…) as second-tier candidates, inline
   compare-template eval (`if out != f"Total: {n}":`) as last resort. All three former guaranteed-exit-43
   shapes now echo-verify; post-run-derived expected still 43 by construction.
3. **Selftest skips counted** — `_record_selftest_skip` ledgers `selfplay_selftest_skip` (reason+detail) on all
   four skip/INCONCLUSIVE paths → visible in `introspect action='activity'` / learning-health.
4. **Calibration honesty** — `FittedParams.map_status` (applied|rejected_inverted|rejected_step|
   discarded_worse) persisted + threaded into refit emit (`refit=map_rejected_inverted`, was `refit=ok` in the
   same cycle the map was REJECTED), boot line, activity ledger, `stats()`.
5. **"No JSON twin" was a wrong lookup** — vector-dedup bump now retries under the STORED duplicate's own
   trigger (metadata → SITUATION-line fallback); TRUE orphans self-heal (delete + write fresh, was: permanent
   veto). Plus `reconcile_vector_orphans` bulk sweep on the skills-auto idle cooldown (token-subset twin match;
   never deletes unidentifiable entries). Contract change: `learn_lesson` returns "written" on the heal path
   (two legacy tests updated to the new contract).
6. **Generator anti-collapse** — negative examples become `filename — gist` fingerprints (quoting 12 full
   same-skeleton openers was in-context REINFORCEMENT of the banned shape — 0.62–0.80 rejects fired WITH the
   steer); mandatory SHAPE-ROTATION steer when ≥½ of the window shares the single-file→aggregate→print
   skeleton; near-dup reject feedback names the shared tokens as BANNED for the retry.
7. **REM churn** — entry/pool-thin announcements moved AFTER the freshness gate (skip ticks print one line, was
   3); hook `_dream_skip_streak` backoff stretches the cooldown 30→60→90→120 min (capped ×4, reset on any
   productive cycle); skip ticks no longer ledger "REM cycle ran".
8. **Warmup verifiability** — warmup + live `Prefill Cache` lines both log the SYSTEM-SLOT sha1 (`sys h=`);
   equal hashes prove the warmed head is the live bytes (they previously measured different segments in
   different units — the 22284-token vs len=8840 "mismatch" in the log was apples-vs-oranges, now checkable).
9. **Quoted-marker mute** — thinking that QUOTED "Emit EXACTLY ONE `<tool_call>`…" latched `stop_printing` and
   silently dropped the rest of the turn's thinking display (the mid-sentence truncations at backticks in the
   log). `_tail_has_stop_marker` now skips markers immediately preceded by a backtick/quote (mention, not
   transition); real transitions still latch.
10. **Trivial tool-routing floor** — bare topic→tool rules ("When asked about the weather, use the
    system_utility tool.") rejected at the lesson-quality gate (`_TRIVIAL_TOOL_ROUTING_RE`); qualified rules
    still pass.
11. **Subjective-gloss verifier rule** (+ cosmetic `Targeting cluster 'None'` fix) — both cheap-judge rubrics
    now state a qualitative gloss of evidenced data ("warm and clear" for 27°C/0% cloud) is supported unless it
    CONTRADICTS the evidence; kills the false-refute class that cost ~24 s of a 43 s weather turn on
    escalation.

Docs updated: `docs/audit_fixes.html` (Round 14), `docs/core/dream.html` (instrumenter, diversity guard,
short-circuit, REM), `docs/memory/skills.html` (twin lookup/orphan heal, routing floor),
`docs/core/agent.html` (stop-marker), `docs/core/calibration.html` (map_status), `docs/core/llm.html`
(warmup hash), `docs/core/verifier.html` (subjective gloss), `docs/memory/frontier.html` (fingerprints).

Still OPEN from the same audit (logged, not fixed here): the confidence composite still has no discriminative
feature (map rejection is the guard working, not the cure — see §4 confidence-feature work); the 20/80
saturation coin-flip means frontier rotation rarely happens while LLM-gen mode-collapses (mitigated by #6, not
removed); native tool_call merge corruption still occurs upstream (repair guard load-bearing).

### 2026-07-29 (later 2) — Night-log audit: the drain that never acked + the validator that failed every correct answer

Overnight-log audit (18:02 → 07:50) found two silent learning-substrate corrupters; both root-caused, FIXED, tested, data scrubbed.

**Journal replay on every restart.** `process_journal_queue` never called `journal.ack()` — the 2026-07-22 TAKE/ACK design relied on "rotation on next pop_all" as the implicit ack, but the idle loop gates the drain on `pending_count() > 0`, so after the last drain of a busy period no take ever rotated the staging file. A COMPLETED batch sat in `memory_journal.inflight.json` indefinitely and `recover_inflight()` replayed it at every boot: the six deploy restarts (00:04–00:55, all clean kills) re-consolidated the same items up to 6× (batch counts 3→4→9→10→11→14), re-burning ~90 s LLM per item and re-teeing post_mortems into the self-play stash. **Fix:** the drain acks per item on every terminal disposition (consolidated / dropped-at-retry-cap / non-retryable error); re-queued items keep their staged copy as the crash backstop. Mid-drain kill now replays only unprocessed items. Tests: `test_smart_memory_requeue.py` (TAKE/ACK section). Stale 01:17 consolidated inflight batch cleared.

**Self-play validator failed 3/3 correct solutions.** 06:49 run: solver printed the exact expected 10 lines (exit 0) yet the LLM-generated validator said "Expected 10 lines, got 1 lines" every attempt — recovered source (counterfactual stash `8d3567930273`) shows `result.stdout.split('\\n')`: a line-split on a LITERAL backslash-n that can never match real newlines. Four gates missed it: static gate had no split lint; echo self-test probe skipped silently (expected built inside `def validate()` → out of scope at the module-level insertion point → exit 43, no log); echo verdict rejected only CRASHES, not clean non-zero exits; reference gate skipped (no reference solution + the omission fail-closed guard needs literal filenames). Cost: false `passed=false, delta=-1.0` on python_general + a wrong solver self-diagnosis. **Fixes:** (1) echo gate now rejects ANY non-zero exit on the validator's own expected output; (2) every gate skip logs a WARNING naming the consequence; (3) static lint `_has_literal_backslash_split` rejects literal-backslash splits at generation (catches the real validator); (4) solver backstop `_feedback_shows_joined_actual` routes actual=='\n'.join(expected) rejections to `validator_infra_crash` (nothing charged); (5) `expected_output_lines` added to probe var names. Tests: `test_selfplay_validator_gates.py`. **Scrub:** frontier run + recent_outcomes entry removed, cluster runs 69→68, `last_length`/`last_compression`/`last_cluster_run_at` restored to post-05:35 state, counterfactual FAILURE entry removed (backups: `*.pre_scrub_20260729`).

**Fresh-eyes review of these same changes found 6 more defects — all fixed before deploy** (the [[review-your-own-changes]] pattern earning its keep again, 3rd time):
1. **`ack([item])` deleted byte-identical TWINS** (`_dedup_key` is pure content; `append` doesn't dedup). Batch `[A, A]`: acking the first cleared BOTH staged rows, so the second twin's ~90 s consolidation ran with no staging record — a kill there lost it outright, i.e. my "fix" introduced a worse loss mode than the bug. `ack` is now **count-aware** (each acked occurrence removes exactly one staged occurrence).
2. **Reject-on-any-non-zero echo exit false-rejected winnable challenges.** The probe echoes the expected variable VERBATIM, so a validator legitimately requiring SHAPED stdout (`FOUND=blob3.txt` — a real `challenge_templates` shape) fails the echo by construction; the strict gate would have forfeited that idle slot. Now rejects only on EVIDENCE — crash in the validator's own frame, or the joined-actual signature — and logs anything else INCONCLUSIVE + allows it through (keeps the gate's "false negatives > false positives" doctrine).
3. Same gate: the probe always strips one trailing newline, so exact-equality validators (`expected` ending in `\n`) failed their own echo → same false-reject class, same fix.
4. **Skip warnings fired on every template cycle** (templates ship without a reference BY DESIGN). New `_pre_verified_shape` flag (template or journal-mined) exempts them from all three new warnings — otherwise pure alarm fatigue on the most common non-LLM path.
5. **Test fixture wasn't faithful to the real validator** (dropped the two list prints), so the backstop was pinned only against a hand-written shape. Fixture corrected + an END-TO-END test now RUNS the recovered validator against a correct solution and feeds its real stdout to the detector.
6. **The lint matched inside comments and its reason quoted the anti-pattern back into the regen prompt** — a model echoing the constraint as a comment would be rejected again, burning every attempt. Lint now scans comment-stripped (string-preserving) source; the reason describes the fix without quoting the pattern. Also now catches the raw-string form `r"\n"` and exempts `re.split("\\n", …)` (the regex engine reads that escape as a real newline).

Suite: 9776 passed / 13 skipped. Known-unfixed (pre-existing, noted): if the journal constructor's `recover_inflight()` raises, `_recovered` stays False and a concurrent `append` during a live drain can fold the in-flight batch back mid-processing.

Also logged from the same audit (not yet fixed): finalize-failed-then-late-CONFIRMED outcome labels as a candidate root cause of the all-night anti-correlated calibration refits (slope −0.7…−0.9); pre-flight guard lacks a world-changed reset (blocked a `manage_services` retry after the agent freed the port, 19:04); vision "node offline" 00:19–00:32 was llama-server failing to decode WEBP with ffprobe missing from the daemon PATH (agent self-healed via PNG conversion).

### 2026-07-29 — Ahmia was never JS-only: a followed 302 + a one-word health check = a confidently wrong diagnosis

Operator pasted a request log where `darkweb_search` returned ZERO results twice (burning a strike and 83 s)
while every Ahmia attempt logged *"HTTP 200 but 0 parseable results — engine serves a JS-only page (no non-JS
results; set GHOST_ONION_ENGINES to a working engine)"*. That message was wrong, and it had been wrong since
the 2026-07-26b bug hunt wrote it into the code AND these docs.

**What is actually true (measured live, clearnet + onion):** Ahmia gates its search endpoint behind a hidden
form field. `/search/?q=…` **302-redirects to `/`** unless the query string also carries
`<input type="hidden" name="2e636d" value="e36dd8">` scraped from its search form. Not Tor-, exit-, UA-,
cookie-, or referer-dependent — purely the missing parameter. With it: 200, ~1.1 MB, ~2900 onion links.

**Why the wrong diagnosis was so convincing.** Two independent weaknesses lined up. (1) `_fetch_raw_html`
follows redirects, so status alone can't tell you the body came from the URL you asked for — a dead endpoint
that 302s to a live homepage looks exactly like a live endpoint serving an unparseable page. (2) The empty-body
branch tested for the bare substring `"javascript"`, which nearly every page contains; Ahmia's homepage carries
a `display:none` notice reading *"we have not deployd non-JavaScript version of Ahmia yet"* aimed at
non-Tor-Browser visitors. The 07-26b hunt's own evidence — "live-confirmed a 4727-byte page with zero result
links" — is the homepage byte-for-byte. It never looked at the redirect chain.

**Fix.** Engine entries may declare `form_token_from`; both Ahmia entries do. `_form_token` scrapes the pair
(`_parse_form_token`, preferring a search-looking form so an unrelated CSRF field can't be mistaken for it),
caches it 30 min, fails OPEN, and runs under a tighter 12 s budget than the search since it spends the same
per-engine deadline. The token is stateless (no cookie), which is what makes cross-circuit caching sound; it is
scraped rather than hard-coded because rotation is the whole point of such a token, and a still-redirected
query drops the cached pair so the retry re-scrapes. `_fetch_raw_html` gained a `meta` out-param carrying
`final_url`, and `_diagnose_empty_body` ranks explanations most-specific-first: redirected-away → an explicit
needs-JavaScript sentence (`_JS_ONLY_RE`, anchored phrasing, no bare word, and deliberately NOT plain
`<noscript>` — working pages wrap analytics pixels in one) → parser drift. **The ordering is the fix**: the
signals are not exclusive, and a page we were redirected *to* also contains whatever notices its author wrote.
Redirect messages name the path only, never the full onion address (operator monitors this stream).

**TWO REVIEW AGENTS CAUGHT THAT THE FIRST FIX DIDN'T ACTUALLY KILL THE BUG — it relocated it.** Both
independently found, and I then verified against a live page, that Ahmia's no-JS banner lives in its
**site-wide template**, not just the homepage. So a healthy ZERO-HIT search (token accepted, no redirect,
engine fine) still matched the JS wording and still told the operator to swap the engine. Three defects behind
it: (a) NO branch existed for the most common cause — the query simply had no matches; (b) the wording test was
**polarity-blind**, matching "works without JavaScript" and "No JavaScript required", i.e. exactly the boasts
privacy-focused onion engines put in their footers — the false positive aimed at the population being scanned;
(c) the sentence was allowed to carry a conclusion it can't support alone. Now: a `no-hits` branch
(`_NO_HITS_RE`) ranked above js-only, polarity-correct patterns, and js-only additionally requires the body to
be under 32 KB (a real JS-only page is a SHELL; Ahmia's results page is 1.1 MB). Live-verified: a real zero-hit
Ahmia page now diagnoses `no-hits`, not "swap engines". **Lesson: tightening a heuristic ≠ bounding what it is
allowed to conclude.** Also from the same review: `_redirected_away` over-fired on www-stripping/default-port/
case (each false positive evicts a good token AND buys a wasted retry every query) → host normalisation; token
eviction now gated on having actually SENT a token and on attempt 0; `_apply_form_token` puts the token before
any `#fragment` (after one it is never sent) and skips a name already in the query (a duplicate `q=` would make
frameworks taking the LAST value search for nothing — which reads as "no hits", so it would never be
invalidated); redirect messages now name host+path, not path alone (a foreign host and this site's homepage
need different responses; onion hosts are scrubbed by the log redactor anyway); and the per-engine deadline is
extended by the token budget for token engines, since 12s token + 30s search = 42s > the 38s deadline would
have guillotined cold-cache searches about to succeed.

**Token behaviour, measured rather than assumed:** the pair ROTATES (it changed mid-session, 2e636d/e36dd8 →
3d0920/2ae6a6) — so hard-coding it would have broken within hours, and scraping was load-bearing, not
defensive. Old pairs keep working for at least hours, which is what makes a 30-min cache safe. Ahmia validates
the exact name AND value: `&foo=bar` and `&3d0920=wrongvalue` both still 302, only the real pair returns
results. The pair is site-wide and identical across cookie jars, transports and circuits, so it is NOT a
per-visitor correlator — it does not link the per-query Tor identities anonymous mode creates.

**Self-review caught one more (the 07-27 lesson again):** the first cut cached the token result unconditionally,
so a Tor blip on the homepage fetch would be remembered as "this engine needs no token" for the full 30 min TTL
— one transient failure stranding Ahmia on the un-tokened path, i.e. 30 minutes of zero results, the exact
failure the token exists to end. Now only a SUCCESSFUL read is cached; non-200/timeout returns None uncached
and logs it. Never record a failed measurement as a neutral value.

**Live-validated:** the exact query from the operator's log — "anonymous communications" — went 0 → **354
results** through the real `_query_engine` over Tor; `ahmia-onion` returned 932 on a second query. Full suite green. New: `tests/test_darkweb_form_token.py`
(21 tests, incl. the ordering regression: redirected + no-JS notice must report the redirect). Docs:
`tools/darkweb_search.html`, including a correction notice on the 07-26b section rather than a silent rewrite.

**Generalisable:** check the redirect chain before believing an engine-health diagnosis, and never anchor a
health check on a substring ordinary pages contain. Also — the log line that pointed at the wrong fix was
itself the artifact of a previous "improve the diagnostics" pass, so a confident diagnostic is not evidence.

### 2026-07-29 (later) — Torch wasn't dead either, it had MOVED: engine survey, 1 index → 3

Operator: "is there any alternative to torch? i don't wanna leave it with just ahmia." Torch had been burning
the full 38s deadline on every search while contributing nothing. Probed the configured `torchdeed…` address
across fresh circuits: **0/10, every one a 30s timeout** — the haystak signature. But the service is ALIVE at
`xmh57jrknzkhv6y3ls3ubitzfqnkrwxhopf5aygthi7d6rplyvk3noyd.onion` under its **Xapian Omega CGI path**
`/cgi-bin/omega/omega?P={q}`. The old `/search?query=` path 404s *there* — so at the right address with the
wrong path it would STILL have looked dead. That is the second "looks dead, actually moved" in one day.
`&HITSPERPAGE=100` takes one page from 7 unique onions to 28 (parser de-dupes by host). Measured 4/4 at
1.6-6.6s — now the FASTEST engine in the set.

Added **torgle** (`no6m4wz…onion/search.php?term={q}`, 4/4, 19-20 results, 9-25s) as a third INDEPENDENT index
— its value isn't size, it's that corroboration ranking had nothing left to corroborate against once torch
died. Measured and rejected: tordex 1/6 (one 86-result hit, else timeout — too flaky to pay a deadline for),
haystak/onionland/tor66 0/2 timeout, phobos 0/2 SOCKS-no-descriptor. Rejected addresses are recorded in a
comment beside the registry so the next person re-measures instead of re-discovering.

Every candidate was measured with THIS MODULE's own `_fetch_raw_html` + `_parse_onion_results` +
`_diagnose_empty_body` over live Tor on rotated circuits — a directory page claiming an engine is up is not
evidence that our fetch and our parser can get results out of it. Net: 1 live index → 3, and a typical search
went from **38s (dominated by torch's timeout) to 17-28s**. Live: 'anonymous communications' 17.3s with
ahmia+ahmia-onion+torch+torgle all answering. Docs: engine-survey table in tools/darkweb_search.html.

### 2026-07-29 — Web reading was DEAD: sync drain on curl_cffi AsyncSession (+ download-tool siblings)

Operator pasted a single live-log line — `RuntimeWarning: coroutine 'Queue.get' was never awaited` at
helpers.py:325 — that turned out to be the only visible symptom of every page fetch failing. Root cause: the
2026-07-26 streaming-cap rework in `helper_fetch_url_content` copied the sync drain pattern from the darkweb
sibling (`_fetch_raw_html`, correctly sync `Session` in a thread) onto the **Async**Session path. AsyncSession
responses stream through an `asyncio.Queue`, so the sync `iter_content()` yields unawaited `Queue.get()`
coroutines → `buf.extend(coroutine)` → TypeError → every 200 fetch returned "Error reading <url>: can't
extend bytearray with coroutine", burned 3 circuit-rotation retries each, and downstream summarize/distill
stages processed error strings as page content. The warning fires ONCE per process (per-location dedup) and
points at the except handler where GC destroys the coroutine — not at the bug. Live-verified broken before /
fixed after (real fetch via Tor returns page text, `-W error::RuntimeWarning` clean).

Fix: `_drain_curl` (aiter_content + byte cap) with shared `aclose_curl_response()` — `quit_now.set()` FIRST
(write callback then returns CURL_WRITEFUNC_ERROR, aborting the transfer) then `await aclose()`; the sync
`close()` is a silent no-op on async responses (reaps `stream_task`, never set — they set `astream_task`).

Post-fix fresh-agent review of my own edits (the 07-27 lesson, again productive): fetch helper verdict clean,
but the SAME defect class was live in `tool_download_file`'s curl branch — (1) sync `close()` on every
redirect hop, (2) non-200 branch slept 5 s for NEWNYM while the abandoned transfer pumped into the unbounded
response queue in RAM, (3) the >50 MB overflow break never aborted the stream. All three now route through
`aclose_curl_response()` (hops, SSRF-reject, non-200/oversize, `finally` around the write loop).

WHY TESTS WERE GREEN: the shared `make_streaming_resp` mock gave every response a bytes-returning sync
`iter_content` — a faithful mock of the WRONG class. It now models the async surface (aiter_content/aclose/
quit_now) and its sync `iter_content` is a TRIPWIRE (raises), so a sync-drain regression on any async path
fails loudly. New regression tests: helpers (sync-drain tripwire, cap-abort ordering quit_now→aclose,
reject-branch abort) + download (hop close, non-200 abort, overflow abort). Full suite 9700 green. Docs:
tools/search.html + tools/file_system.html. DEPLOYED (supervisor restart).

### 2026-07-27 (later 14) — uncertainty_pressure de-zeroed (ordering bug) + feature-health verdicts made honest

Follow-up to the operator's learning-report evaluation: the two real to-dos it surfaced.

UNCERTAINTY_PRESSURE ≡ 0.0 — ROOT-CAUSED, an ordering bug, not a dead concept. Evidence: the durable
uncertainty_log.jsonl has 2 records EVER (both hedge-scan) and the flag_uncertainty tool has never been called
live. Three arms:
• Non-streamed finalize: _record_calibration_safe reads tracker.pressure() ~20 lines BEFORE the hedge
  auto-scan populated the tracker (the scan lived in the surfacing block, after the record) — every sample
  recorded the pre-scan empty state. The scan — designed to be "load-bearing even when the LLM never calls
  flag_uncertainty" — fed only the user-facing footer, never calibration. Scan hoisted above the record;
  surfacing block keeps footer/verify/reset on the already-populated tracker.
• Streamed path: NO scan ran at all before the end-of-stream pressure read. Scan added on full_content before
  the read; drain now resets the tracker after its calibration record (finalize's reset never runs on the
  stream path — without it, hedge state would leak into the NEXT turn's reading).
• Feeder volume: hedge regex broadened conservatively (unable to verify/confirm, can't confirm, I'm
  uncertain, no way to verify/check/know). Deliberately narrow — false pressure on confident turns poisons λ
  the other way. Expected post-fix shape: sparse-but-real (nonzero only on verbally-hedged turns).

FEATURE-HEALTH VERDICTS: _feature_health blended entropy's neutral 0.5 stand-ins with real observations —
~1300 stand-ins forced separation≈0 by construction and branded entropy DEAD hours after the n_probs fix
started producing real values (same "no signal" vs "neutral measurement" conflation as the
entropy_distinct_values fix). Entropy now judged over OBSERVED rows only; every feature row carries n; verdict
is three-way live/dead/insufficient (<10 eligible rows or one outcome class → withheld, and critically NOT
"live": 5 one-class observations must not read as a working feature).

Tests: tests/test_uncertainty_pressure_wiring.py (new — behavioral hedge→pressure + source-pinned ordering on
both paths + single-scan-site + drain reset) and test_calibration_probability_map.py feature-health cases
(stand-ins→insufficient, observed-minority-not-drowned, tiny-corpus-withheld). Docs: docs/core/uncertainty.html
(ordering-bug section), docs/tools/introspect.html (verdict semantics).
FULL SUITE: 9638 passed, 13 skipped. DEPLOYED (70152→70508); functional_live_test 32/32 (one warm-up soft on
the first run right after restart, clean on rerun). LIVE-VERIFIED the mechanism: a hedged live turn recorded
uncertainty_pressure=0.0667 — the FIRST nonzero in the corpus's history (unhedged turn before it: 0.0; risk
footer rendered, durable log accruing hedges incl. from background turns). Live feature table now honest:
entropy n=14 [insufficient — one outcome class so far], uncertainty_pressure distinct=2 (was 1 forever),
effort live. Watch both via introspect action='learning' as observed failed-class samples accrue.

### 2026-07-27 (later 13) — w_entropy un-pinned: native n_probs sidesteps the tools+stream logprobs 400

Operator asked how to fix the streamed-path logprobs issue (entropy coverage 0/1280, w_entropy pinned).
Diagnosis chain: the earlier same-day hoist fix requests logprobs only when "tools" not in payload — but the
main loop attaches tools on EVERY non-final generation, and most turns END on a tools-attached generation (the
model answers without calling a tool), so nothing was ever observed. The llama-server constraint was then
re-read at the SOURCE (llama.cpp master server-common.cpp / server-task.cpp, matching live b10090):
• the 400 guard fires ONLY on the OAI `logprobs` flag (`json_value(body,"logprobs",false)`);
• the native `n_probs` field rides the server's pass-through parameter copy straight into the sampler;
• the streamed chat formatter attaches OAI-shaped logprobs to a chunk's last delta whenever the slot sampled
  with n_probs > 0 — it never re-checks the request flag.
So `n_probs + tools + stream` streams logprobs. VERIFIED live before writing code: probe against :8088
returned logprobs-bearing chunks that parse through extract_top_logprobs → EntropyTracker unchanged (sparse
under MTP — expected, tracker reads observed tokens). Probs/delta alignment is imperfect under tool parsing
(why upstream guards the strict-OAI path) — irrelevant for entropy, which needs top-K distributions only.

IMPLEMENTATION:
• entropy.request_logprobs(payload, top_k, native_nprobs_ok) owns field selection: no tools → portable OAI
  fields; tools → native n_probs. The OAI flag is NEVER set alongside tools (that 400 breaks the GENERATION).
• agent.py metacog block now calls it (single site, covers both stream paths; tracker arming unchanged).
• Self-healing: if a future upstream rejects n_probs, the stream-abort handlers on BOTH paths latch
  context._nprobs_rejected → later generations fall back to the no-tools-only gate. One broken generation,
  not every one. Kill: GHOST_ENTROPY_TOOLS_NPROBS=0 (e.g. non-llama.cpp upstream that rejects unknown fields).
• learning_health entropy note rewritten: coverage should CLIMB from here; flat 0 = probe broken again.
Known risk, stated: this rides llama.cpp implementation details (pass-through + formatter behavior), which is
why the latch + kill-switch exist and why the wiring test pins the OAI-flag-never-with-tools invariant.

Tests: test_calibration_entropy_wiring_2026_07_27.py rewritten around the new invariant (request_logprobs
behavioral tests: OAI-flag-never-with-tools, n_probs path, rejection latch, env kill, chunk-shape parse;
latch wired on both paths). Docs: docs/core/calibration.html (n_probs sidestep section; the "structurally
sparse" watch flips expectation), docs/core/entropy.html (request_logprobs API row + consumers note).
FULL SUITE: 9626 passed, 13 skipped. DEPLOYED (68659→69009, health ok); functional_live_test 32/32.
LIVE-VERIFIED the mechanism, not the logs: the 3 newest calibration samples (post-deploy live turns) carry
entropy_observed=True with REAL varying entropy (0.999/0.965/0.977 vs the corpus-wide neutral 0.5) — the
first observed entropy on tool-attached turns ever recorded. Watch entropy_observed_pos/neg via
introspect action='learning'; w_entropy unpins once ≥30 observed samples span both outcome classes.

### 2026-07-27 (later 12) — Introspect-subsystem review: 5 bugs (2 instrument≠mechanism) + 8 improvements

Operator asked for a dedicated review of the introspect subsystem (tools/introspect.py, core/learning_health.py,
the selfhood read path, registry wiring). 5 bugs + 8 improvements, all fixed/applied same session.

BUGS FIXED:
• Competence inject-gate mis-mirrored the mechanism (learning_health): the report gated per-domain (n≥20 each)
  and could print "NONE (block not injecting yet)" while agent.py — which gates on the TOTAL observation count
  across domain rollups and then renders EVERY domain — injected the block every turn (e.g. 4 domains at n=8,
  total 32). Now mirrors the real rule (total_observations ≥ gate) and reads the gate from
  GhostAgent._COMPETENCE_MIN_OBS itself. This is the keep/kill instrument — a wrong "not injecting" here
  triggers a wrong verdict at exactly the decision it exists for.
• entropy_learnable used a different formula than the fit: report said distinct≥3 & observed≥30;
  calibration.py's gate is observed ≥ _MIN_ENTROPY_SAMPLES AND both outcome classes among the OBSERVED
  samples. 40 one-class observed samples read "LEARNABLE" while the fit pinned w_entropy=0. Now mirrors the
  fit exactly and reports entropy_observed_pos/neg + the live floor.
• action='learning' was in the enum but ABSENT from the tool description → undiscoverable by the model (tool
  selection is description-driven; it was effectively operator-only). The description's closing "All reads
  route through your SelfModel" was false for activity/learning. Fixed in registry.py; the same stale claim
  sat in docs/tools/registry.html (patch one sibling → grep for the others).
• Activity report silently truncated its window: the tail scan is capped (512KB / 1000 records) but the header
  claims "last N h" up to 336h — and _read_activity_tail's bare `except: pass` rendered a FAILED read as the
  calm "No background activity recorded in the last Xh", instrument failure indistinguishable from a quiet
  fortnight. Now: a truncation note when the oldest scanned record is newer than the window start, and a read
  failure logs + renders as an explicit read error ("not \"nothing ran\"").
• The activity branch returned BEFORE the tool's try-block — it escaped the docstring's never-raises contract
  (upstream gather(return_exceptions=True) saved the turn, but as a raw invocation error). Now guarded in-branch.

IMPROVEMENTS: operating principles rendered in stats (count) + summary (full list) — the "behaviour-shaping"
values layer was invisible to introspection; recent/recall lines age-stamped ("3.2h ago", same time language
as the activity report); AutobiographicalMemory.count()/cluster_counts() served from the (mtime,size)-cached
search index — stats() previously cost two whole-log scans per call on the summary path; every mirrored gate
constant now imported LIVE from its owning module (agent / calibration / memory.skills — skills' stale gates
got named constants _STALE_MIN_RETRIEVALS/_STALE_HIT_RATE) with last-known fallbacks; _load_jsonl bounded via
deque(maxlen=limit) and honours the limit on mid-read failure; pass/fail/mixed lesson bucketing single O(n)
pass (was O(n²) dict-equality membership, misclassified duplicate lessons); pretty_log on every introspect
action (only summary/activity announced themselves before); tests for action='learning' end-to-end + all of
the above.

The recurring theme, again: instruments disagreeing with their mechanism (both gate bugs were hand-copied
mirrors that drifted) and reports that render their own failure as "quiet". Constants now flow FROM the
mechanism; failure states are worded as failures.

Tests: test_learning_health.py grew the divergence cases (per-domain vs total, one-class entropy corpus +
mechanism-constant assertions); test_selfhood_introspect_tool.py grew learning/activity-guard/truncation/
read-error/principles/age/cache-invalidation coverage; tail-read tuple signature updated in
test_bughunt_fixes_2026_07_27.py. FULL SUITE: 9621 passed, 13 skipped (3:48).
Docs: docs/tools/introspect.html (actions/params tables now include activity+learning, failure semantics,
full review section), docs/tools/registry.html (stale introspect blurb corrected).
DEPLOYED (65219→65742, health ok); functional_live_test 32/32 (0 soft). Live-verified the corrected
instrument against production stores: COMPETENCE "INJECTING (2630 total obs ≥ 20-total gate)" and entropy
"0/1280 observed (0+/0-) → pinned, COVERAGE not degeneracy" — both lines now state the mechanism's truth.

### 2026-07-27 (later 11) — Auth-rejection log noise: re-levelled, never suppressed

Operator spotted `auth rejected path=/api/health` ×2, then `path=/api/game/move`. Traced: the agent's OWN
`functional_live_test.py` deliberately probes with a missing key and a wrong key to prove auth is enforced, so
every run emitted WARNING lines indistinguishable from a real intruder. That is how a security signal gets
learned-ignored — the noise was ~2-3 lines per run, on the one log line that should always mean something.

FIXED without creating a suppression mechanism. The obvious approach — downgrade when the User-Agent says
"it's the test" — is WRONG: headers are attacker-controlled, so it hands anyone a switch to mute their own
probes. The rule is therefore BOTH conditions, and re-levelling only:
• the marker `ghost-functional-test` is honoured ONLY from LOOPBACK (an attacker must already be on this host);
• the line is ALWAYS emitted — only WARNING→INFO changes. Nothing can be made to disappear;
• the 403 is untouched;
• `ip` and `ua` are now always logged, so a real hit stays identifiable instead of being a bare path.
Residual risk, stated rather than hidden: a local non-owner user could lower their probes to INFO. They remain
logged, and anyone with local access is already past the boundary the key protects.

One iteration mattered: my first version appended `[own functional suite]` at the END of the message, and the
log's width cap TRUNCATED it away — leaving the one field that says "self-inflicted" invisible, i.e. the whole
point of the change silently lost. Tag moved to the FRONT.

Tests: test_auth_rejection_logging.py (18) — pinning that a bad key is always 403, that a line is always
emitted, that key bytes never appear, and specifically that the marker from a REMOTE host stays WARNING.
Ran the auth/API-relevant suites only (186 passed) rather than the full 9602 — a log-format change does not
warrant a 4-minute full run (operator's call, and correct).
DEPLOYED (57358→57574, health ok); functional_live_test 32/32.
LIVE-VERIFIED both branches:
  `auth rejected  [own functional suite] path=/api/health ip=127.0.0.1`   ← INFO, self-inflicted
  `auth rejected  path=/api/game/move ip=127.0.0.1 ua=curl/8.0-intruder`  ← WARNING, real shape

### 2026-07-27 (later 10) — Tier 2 REDESIGNED from evidence: failure reports, not correction phrases

Operator: "what do we need to proceed with tiers 2-4?" → I measured the SUPPLY of each before writing code,
which killed my own design. Then "run the scan" → "proceed".

**THE SCAN THAT OVERTURNED THE DESIGN.** Over the stored sessions (56 sessions, 246 eligible triples after
excluding 23 single-turn ones): the live correction classifier fires on **0**. Not one, ever. But 20 of those
246 (8.1%) unmistakably report the delivered work is broken. So the question "does the operator rarely correct,
or can the classifier not see it?" resolved decisively to the SECOND — and my proposed Tier 2 (loosen
phrase+rephrase to phrase-only) would have caught NOTHING, because the phrase is exactly what's absent:
    "game.js:45 Failed to load frame definitions: TypeError: …"
    "Failed to load resource: the server responded with a status of 404"
    "it still does the same. the game never starts"
A pasted traceback contains no rebuttal phrase and shares no tokens with "build me a game", so neither of the
classifier's two signals can fire. It is not malfunctioning; it is structurally blind to how corrections
actually arrive here (~1/day of GROUND TRUTH, the largest untapped negative source in the system).
**ALSO KILLED: "repeat request = implicit negative"**, which I had proposed two messages earlier. 11 near-
identical re-asks exist and they are dominated by "hello ghost, what's new today?" — a daily-briefing habit.
Treating those as negatives would have poisoned the corpus with fabricated failures.

SHIPPED — `classify_failure_report()` (distill/user_correction.py), single-signal by design (a pasted traceback
is unambiguous in a way a bare "no" is not):
• DIAGNOSTIC: tracebacks, exception classes, HTTP 4xx/5xx, "Failed to load", "is not a function", file.js:NN.
  Bare "error"/"errors" deliberately EXCLUDED — it false-positives on "zero console errors"/"no errors now".
• BREAKAGE, split by STRENGTH — this split is the load-bearing part:
  - STRONG ("doesn't work", "still does the same", "won't start", "nothing happens", "needs manual reload")
  - WEAK ("fix it/that/this") — ambiguous: a complaint in "doesn't work, please fix it", but forward-looking
    instruction in "perfect, fix it exactly like that next time".
• PRAISE VETO scoped to WEAK evidence only. My first version vetoed on any affirmation, and it silently ate
  "the minesweeper right click doesn't work though, but the rest looks good" — a real defect report with a
  polite softener. **Mixed messages are the COMMON shape**; an over-aggressive guard discarding real signal is
  the same class of bug as the tolerant parser that inverted a verdict earlier today. A softener does not
  cancel a named breakage; a hard diagnostic outranks any amount of politeness.
RESULT on the real corpus: **20/246 detected (8.1%), 0 false positives, 0 vetoes needed** — the veto guards
only hypotheticals, which is what a guard should do.

WIRED calibration-ONLY, deliberately: unlike an explicit correction this does NOT mark the trajectory FAILED,
because that feeds reflection → lessons → playbook where a false positive poisons retrieval (the module's own
docstring warns of exactly this). A calibration sample is numeric, provenance-tagged, and filterable back out;
a bad lesson is not. Recorded at `source="failure_report"`, grade 0.15 (a notch above an explicit correction's
0.0 — the human is reliable that something is broken, slightly less so that THIS turn broke it). The
fingerprint stash is consumed, so a report and a correction on one turn can never double-count.

**TWO BUGS FOUND BY THE LIVE TEST — both mine, both invisible to 9584 green tests:**
1. **Tier 2 sat behind the TRAJECTORY-cache lookup.** `traj = cache.get(fp); if traj is None: return` ran BEFORE
   any classification, so a turn whose trajectory was never cached (or was LRU-evicted) returned before Tier 2
   could run — silently disabling it on exactly the long, tool-heavy turns most likely to draw a complaint.
   Moved the lookup AFTER classification; Tier 2 needs only the fingerprint + calibration stash. An uncached
   trajectory now still files the correction negative instead of losing the strongest label available.
2. **An interaction between two of my OWN same-day changes.** The D1 fix (later-6) gated the stash on
   `_calib_outcome >= 1.0`, correct when labels were binary. The graded label (later-9) scores a clean turn
   0.83 — never 1.0 — so the stash stopped being populated for almost every turn, disabling BOTH the
   user-correction negative AND Tier 2. Threshold → 0.5. **Neither change is wrong alone; the composition is.**

**INCIDENT (self-inflicted, resolved):** rapid deploy cycling raced — a new process could not bind :8000 while
the old one still held it, and it kept running WITHOUT a listener while launchd reported "running". Symptom:
`HTTP 000` connection-refused with a live pid. Fix: kill + wait for an actual bind (took 168s). Lesson: the
deploy check must verify a LISTENER, not process liveness — `launchctl print` saying "running" is not service.
Also noted: a synthetic probe citing a nonexistent `app.js:12` sent the agent on a 20-turn/856s hunt. Probes
for this tier must bound the agent's response ("do not investigate, reply exactly X").

Tests: test_failure_report_tier.py (45) — the corpus is the 20 VERBATIM session messages plus the real praise
traps, so it is a regression corpus rather than invented examples. One pinned test updated (the 1.0→0.5 gate).
FULL SUITE 9584 passed / 0 failed. DEPLOYED (54109→55713 after the incident, health ok); functional_live_test
32/32. LIVE-VERIFIED: `outcome=0.15 source=failure_report`.
REMAINING: §4E Tier 3 (reopened work, ~7 events/20d) and Tier 4 (HOLD — needs a different instrument).

### 2026-07-27 (later 9) — Graded outcome labels + sample provenance (negative-supply Tier 1)

Operator asked how I'd widen the negative-label supply, then "proceed". Shipped the two pieces I proposed to
do FIRST — provenance, then grading — deliberately stopping before the implicit-signal tiers so the effect of
this change is attributable on its own.

**The reframing that drove it:** "too few negatives" (49 of 1226) is the visible problem, but the BINDING one is
that the label was near-constant AND measured the wrong thing. `0.0 if (exec failure | verifier REFUTED | budget
exhausted) else 1.0` asks "did anything visibly break", not "was the answer good" — so a turn that hit one tool
error, RECOVERED, and answered correctly scored 0.0, identical to a refuted answer. That is a mislabel, not a
harsh label.

SHIPPED:
• **`source` on every calibration sample** (`turn` = the graded proxy, `user_correction` = ground truth), added
  BEFORE any new signal tier. Without provenance, mixing tiers is irreversible: you can never audit which tier
  is noisy, nor drop one, without discarding the whole corpus. Legacy rows default to `turn` (what they were).
• **Graded label in [0,1]** (`grade_turn_outcome`) replacing the binary one. Constants are MEASURED, not
  chosen: across 302 verdict-bearing trajectories the agent passed 251, so an unverified-but-clean turn scores
  **0.83 = the observed P(good | checkable)** rather than an asserted 1.0 — asserting 1.0 for a turn nothing
  checked is the verification theatre this project keeps rediscovering. Verifier PASSED/FAILED stay the hard
  1.0/0.0 anchors; exec failures subtract 0.15 each to a 0.15 floor (0.0 is reserved for "checked and WRONG");
  budget exhaustion = 0.2 (the agent itself flags the reply PARTIAL).
• **Stopped binarising on BOTH the write and read paths** — `1.0 if outcome >= 0.5 else 0.0` would have crushed
  every graded label straight back to two values, i.e. re-created the exact constant column the grading exists
  to remove. Now clamped only.
• **Class-presence gates generalised to VARIANCE** in `fit()` and `_fit_platt`. "Are both binary classes
  present?" is the wrong question once labels are continuous: a graded corpus can carry real signal with every
  sample above 0.5, and the old gate would have refused to fit it at all. For 0/1 labels the two tests are
  identical (variance is 0 exactly when one class is missing), so binary corpora behave bit-for-bit as before.
• Telemetry: label variance / distinct values / mean / per-source counts, plus a warning when the label is
  binary-or-flat. Fixed my own wording bug: it reported "LOSES TO the base-rate predictor" when the two were
  EQUAL — equality is a tie, not a regression.

**MEASURED, AND IT CONTRADICTS MY HYPOTHESIS — recorded honestly.** Replaying the real production mix (1028
turns, 726 unverified) through the real feature + real fit:
| label | distinct | variance | class split | brier vs base |
|---|---|---|---|---|
| BINARY | 2 | 0.0986 | 914/114 | 0.0666 vs 0.0986 → **beats by 33%** |
| GRADED | 8 | 0.0446 | 969/59 | 0.0395 vs 0.0446 → **beats by 11%** |
So grading gives a far richer target (8 values vs 2) but a SMALLER edge over the baseline. Why: the binary
label is partly CIRCULAR with the effort feature — effort counts tool calls/repeats, tool errors drive both the
effort signal and the binary label, so the model was partly predicting itself. Grading weakens that
near-tautology, which lowers the apparent win. The 33% was flattering; 11% is the honest number.
I shipped it anyway because the binary label is factually WRONG about recoveries, and a correct label with a
smaller flattering-metric is worth more than an incorrect one with a bigger one. Noting the trade explicitly so
nobody later reads the drop as a regression.
Caveat also recorded in-code: the graded label is a PROXY for process health, not correctness. If the
ground-truth tier (user corrections) ever stops flowing, the agent is calibrating purely against its own notion
of a tidy turn — which is exactly what it already over-indexes on. Provenance is what keeps that checkable.

LIVE-VERIFIED: unverified chat turn → `outcome=0.83 source=turn`; verified tool turn → `outcome=1.0`.
Telemetry: `labels: variance 0.03797 over 3 distinct values (mean 0.9603) · sources turn=1240`.
One pre-existing test updated (it pinned the binarisation by name). Tests: test_calibration_graded_labels.py (18).
FULL SUITE 9539 passed / 0 failed. DEPLOYED (49604→50192, health ok); functional_live_test 32/32.
NEXT: **Tiers 2-4 are logged as PENDING in §4E** (designed, ranked, with the two standing hazards — proxy drift
and label leakage — written down). Not started deliberately: measure Tier 1's effect on live data first, or a
second signal source makes attribution ambiguous.

### 2026-07-27 (later 8) — Operator log report: news_headlines invisible + the prefix cache primed twice

Operator: "1. something is wrong with kv cache, it seems to be loading ~24k tokens twice. 2. news_headlines
custom skill stopped working." BOTH ROOT-CAUSED FROM THE LOG — and they turned out to be the SAME root cause
wearing two hats: the advertised acquired-skill set.

**#2 news_headlines — the skill was INVISIBLE, not broken.** The log's tell was one line repeated on EVERY
request regardless of query: `semantic routing injected 1: format_results_to_csv` — even on the request whose
hydration was literally "news_headlines tool usage". Measured the stores: **registry has 3 active skills, the
vector store had 1 embedding.** Routing queries the vector store (`where type=acquired_skill`, n_results=15) and
filters the advertised tool schema by the result, so news_headlines was never in the catalogue the model saw.
The whole failure cascade in the log follows from that: model calls `manage_skills` twice looking for it →
tries to shell out (`inline script blocked — body 219 chars >= 120`) → `loop breaker: 'manage_skills' repeated
2x` → forced-final turn → `Scrub consumed entire response (intended=news_headlines); emitted fallback`.
WHY IT COULD NEVER SELF-HEAL: `save_skill` embeds ONLY when the content hash CHANGES, so once an embedding was
lost (the pre-07-26 broken deletes, or an orphan sweep) nothing ever rebuilt it. The index could only LOSE
entries. The skill stayed `active` and dispatchable while being invisible — "invisible-but-callable drift"
arrived at from the opposite side to the one the routing degrade-paths were written to guard.
FIXED: `backfill_missing_skill_embeddings()` re-embeds any ACTIVE registry skill with no vector entry, wired
into the same sweep as the orphan purge. Also HARDENED the purge: an empty registry alongside existing
embeddings is now treated as a failed read, not as "all skills deleted" — without that guard one transient
mid-write read wipes the entire routing index, unrecoverably.
LIVE REPAIR: backfilled 2 (news_headlines, generate_password); a news query now ranks news_headlines FIRST.

**#1 KV cache loading ~24k twice — same cause.** The boot warmup prefills the request head via
`get_active_tool_defs("")`. Empty string is FALSY, so routing is skipped and it advertises ALL skills. Every
live request passes a real query → routing → a SUBSET. So the ~22186-token warmup prefix never matched a single
real request, and each request re-prefilled the whole head from scratch — the head was loaded twice. The
warmup's own comment claimed it used "the same neutral-query routing a live request's first turn resolves to";
that was false the moment routing became query-dependent.
FIXED at the design level rather than by patching the warmup: below `_SKILL_ROUTING_MIN_SKILLS` (25) the
registry advertises EVERY active skill and skips per-query routing entirely. Filtering 3 skills saves a few
hundred tokens once and costs a full re-prefill of a 22k head on every request — a terrible trade. Above the
threshold routing still applies. This also makes the tool block byte-identical ACROSS requests, not just across
the turns of one request (the per-turn stability was already correct — F3 held h=28cc7eb2 over 4 turns).
VERIFIED LIVE: the "semantic routing injected" line is now ABSENT on live requests (= advertise-all = the same
set the warmup primed), and `news_headlines` dispatches for real: `turn outcome verified · tools:
news_headlines · 804 chars`, returning actual Greek RSS headlines. No scrub fallback.

Three pre-existing tests updated (they assert routing FIRES; they now monkeypatch the threshold down, since
their fixtures hold 0-3 skills). Tests: test_skill_index_and_prefix_cache.py (10).
FULL SUITE 9521 passed / 0 failed. DEPLOYED (46905→47701, health ok); functional_live_test 32/32.
LESSON: a log line that is IDENTICAL across every request regardless of input is not background noise — it is a
constant where a variable belongs. That one line named both bugs.

### 2026-07-27 (later 7) — Second review pass: 6 MORE defects in the same day's calibration work

The re-review of `confidence.py`/`calibration.py` (the first attempt died on an API error) found six more,
ALL introduced earlier the same session and ALL invisible to a 142-green calibration suite — each needs a corpus
shape the tests never built (linearly separable, single-valued, or a fit that actually gets adopted).
THEME: **guards written from intuition instead of measurement.** Every one of my safety mechanisms was wrong in
a way that only a numeric probe reveals.

**D1 — the "ridge" was not a regulariser.** I added it to the Hessian diagonal only, which is Levenberg damping:
it changes the STEP SIZE but not the fixed point. On linearly separable data the unpenalised optimum is at
infinity, so the slope just grew with the iteration budget — a = 51 → 207 → 233 at 5 → 50 → 200 iters —
directly falsifying my own docstring guarantee that the answer must not depend on the budget. FIXED: the L2
penalty now enters the GRADIENT as well (`g_a += ridge*a`), making it a true MAP estimate; ridge 1e-6 → 1e-2.
Verified budget-independent: a = 20.76 at 5, 50 and 200 iterations.
**D2 — and that diverged map was then ADOPTED**, because `brier_cal <= brier_raw` is computed IN-SAMPLE on the
same points the 2-parameter map was fitted to, so it is essentially always true. Result: confidence collapsed to
the step predicate `competence < 0.5499`, reporting a perfect Brier 0.0. FIXED via a `_MAX_SLOPE = 50` backstop.
**D3 — my `_MIN_SLOPE = 0.5` floor was scale-blind AND justified by a false claim.** I wrote that a flat map
"would leave the gate inert"; measured, that is wrong — Platt is strictly monotone for a>0 and τ is refit on the
SAME scale, so the below-threshold decision set is bit-identical (verified: |below| 412 vs 412). Rejecting
a = 0.296 discarded a real 0.069 Brier improvement and changed NO decision. Slope also can't be judged without
the composite's spread (a=3.0 over a 0.02-wide range moves probabilities less than a=0.3 over the unit
interval). FIXED: reject only what is genuinely unsafe — INVERSION (a<=0, flips the ordering) and DIVERGENCE
(a>50, near-step). Small positive slopes are kept.
**D4 — `w_effort == 1.0` was silently dropped** (bound was `< 1.0`, but the grid reaches exactly 1.0), leaving
`self.w_effort` at the PREVIOUS fit's value while the other two weights updated — the scorer then evaluated a
formula the fit never saw, with a Platt map fitted on different composites. FIXED to `<= 1.0`.
**D5 — the map's intercept FLOORED the composite so a REFUTED verdict could never cross the threshold.** I
applied Platt after the outcome penalty, making `sigmoid(b)` an absolute floor: with a fitted b = 0.27, even
`outcome_penalty=1.0` bottomed out at 0.566 and `below_threshold` was unreachable for EVERY possible input —
while the comment two lines above promised the penalty "must pull the reading below threshold regardless".
FIXED: penalty now applies AFTER the map, on the probability scale, so it always reaches 0 (verified 0.0/below
at penalty=1.0). Also `_best_threshold`'s degenerate fallbacks returned hardcoded 0.5/0.55 — raw-scale constants
handed to a caller on the mapped scale — now the MEDIAN of the supplied scores, which is scale-free.
**D6 — the stored `composite` column mixed scales.** It held raw values for pre-fit turns and Platt-mapped ones
after, re-scaled again at every refit, so the reliability table / ECE / reported Brier silently compared
different populations (same turn binned [0.2,0.3) pre-fit and [0.6,0.7) post-fit). `fit()` was immune (it
recomputes from components) but every diagnostic was not. FIXED: `ConfidenceReading` now carries
`raw_pre_penalty_composite` and BOTH record sites persist that, so the column has one stable meaning.
**D7 — `stats()` mixed windows**: Brier over the max_history tail, ECE over the entire file (measured 0.0478 vs
0.1593 on the same report). FIXED to share the window.

Two pre-existing tests updated to the corrected behaviour (the 0.55 threshold constant is now a scale-free
median; a source-window assertion widened). Tests: test_calibration_review_fixes.py (19).
FULL SUITE 9511 passed / 0 failed. DEPLOYED (45989→46146, health ok); functional_live_test 32/32.
NOTE none of D2/D3/D5 were ACTIVE in prod — the map is currently rejected on live data, so they were latent —
but they would have activated the moment the effort feature earns its weight (~1 day), which is exactly when
nobody would have been looking.

### 2026-07-27 (later 6) — Reviewing the same day's own changes: 4 defects, 3 self-inflicted

Operator: "are we done? do we need to scan the files you just changed? more functional tests?" — YES on both,
and the review paid for itself. Two read-only agents re-reviewed today's confidence/calibration/agent.py edits.
Every defect below is on a path the 9477-green suite never constructed, which is precisely why they survived.

**D4 (WORST — a fix that made things worse). Hypothesis survivor-index coercion inverted the verdict.**
Earlier today I replaced a hard `int(_i)` with a "tolerant" `re.sub(r"\D","",label)`. But stripping ALL
non-digits CONCATENATES separate numbers: `"H2: retry after 30s"` → `"230"` → index 230. A non-empty set is
TRUTHY, so the promote/demote block then ran and set `consistent=False` on every genuine hypothesis — the model
said H2 survived, the code concluded all were refuted. That is strictly worse than the hard failure it replaced
(which at least left the list untouched). FIXED: anchored `re.match(r"\D*(\d+)")` (first digit run only) plus a
range check `0 <= ix < len(hypotheses)`, so an out-of-range or prose label yields NO phantom survivor.
LESSON: making a parser "tolerant" without bounding what it may produce converts a loud failure into a silent
wrong answer. Verify the OUTPUT domain, not just that parsing succeeds.

**D1. Calibration negatives could be double-counted.** The correction stash was written unconditionally,
including for turns already recorded at outcome=0.0 (execution failure / REFUTED / budget exhausted). A later
user correction then recorded a SECOND 0.0 for the same turn — double-weighting it in Brier, ECE and the weight
fit. FIXED: stash only when `_calib_outcome >= 1.0`. The stash exists for turns that looked CLEAN and were
nonetheless wrong; a turn already booked as a failure has nothing to add.

**D2. The stash stored the POST-penalty composite** while the inline record deliberately stores
`pre_penalty_composite` (the penalty is a function of the label, so storing it makes the negative class read
optimistically — a bug the inline path documents and avoids). On a penalised-then-corrected turn the stashed
value was ~5x below the prediction actually made. FIXED to match the inline path.

**D3. The stash omitted `effort_component`/`effort_observed`** — so the fit, which filters on those flags,
excluded correction negatives from the effort-weight fit entirely. Same class as the `entropy_observed` gap I
fixed hours earlier; I patched one sibling and missed the other. FIXED (both flags + the value now carried).

FUNCTIONAL TESTS RUN LIVE (paths changed today that had never been exercised end-to-end):
• **User-correction path** — needed BOTH classifier signals (correction phrase AND a high-Jaccard rephrase of
  the ORIGINAL request; my first attempt had only the phrase and correctly did nothing). With both:
  "trajectory promoted — prior turn marked FAILED via user-correction" and calibration negatives 48→49. ✅
• **Streamed path records turn-effort** — `effort_observed: true`. Note the value was exactly 0.5, which is ALSO
  the unobserved default: 4 identical `file_system` calls → sprawl 0.333 + spin 0.667 → 0.5. A measured 0.5 is
  indistinguishable from the default BY VALUE, which is exactly why the `_observed` flag is the load-bearing
  part of the design. ✅
• **Games** — agent opening as O returns `....O.... X` and replaying it now gives HTTP 200 (used to 422 =
  bricked after one move); blank state 422; count-contradicting turn 422; no-key 403. ✅
• Non-streamed effort + `domain` recording, SQL multi-statement guard, health — all re-verified. ✅

Tests: test_review_fixes_2026_07_27.py (15). FULL SUITE 9492 passed / 0 failed. DEPLOYED (43896→44746, health
ok); functional_live_test 32/32.

### 2026-07-27 (later 5) — Turn-effort: the confidence score finally carries information

Operator: "give confidence a per-turn feature." Prerequisite finding from later-4: ALL THREE inputs were dead,
and competence — the only one that varied — is a per-DOMAIN historical average, so it is near-constant within a
domain and STRUCTURALLY cannot discriminate individual turns (separation −0.0008, leak-free AUC 0.473).

**MEASURED BEFORE BUILDING** (296 labelled trajectories, 246 passed / 50 failed):
| signal | mean(passed) | mean(failed) | AUC |
|---|---|---|---|
| tool calls in turn | 2.39 | 11.70 | 0.334 |
| longest same-tool run | 1.35 | 5.82 | 0.321 |
| distinct tools | 1.39 | 2.08 | 0.377 |
| *any tool errored* | *0.10* | *0.64* | *0.229* ← **EXCLUDED, see below** |

**`has_tool_error` deliberately NOT used** despite being the strongest correlate: it is literally a term of the
calibration label (`execution_failure_count > 0`), so it would predict the label FROM the label and add zero
information — the same circularity that made later-4's AUC 0.679 an artefact. Ruling it out was the main design
decision here.

SHIPPED — `confidence.effort_component(tool_names)`: averages two saturating struggle signals, SPRAWL (call
count, saturates at 12) and SPIN (longest same-tool run, saturates at 6), into [0,1] where 1.0 = short clean
turn. Averaged not min()'d so one extreme can't slam it to 0; measured AUC 0.670 vs 0.678 for the best single
signal, i.e. within noise while depending on both. Wired on BOTH scoring paths (finalize + streamed).
**END-TO-END RESULT** (replaying the labelled corpus through the real feature + real fit): `w_effort` earned at
0.5, Platt map ADOPTED (a=5.91 — passes the slope guard that rejected it on the old featureless data),
**Brier 0.123 vs base rate 0.140 → BEATS the baseline by 12%, where it previously LOST by 75%.** First time the
confidence score adds information.

Design rules carried from the earlier fixes (all three were load-bearing):
• **No fabricated neutrals** — a turn with no tools passes `effort=None`, not 0.5. Absence of evidence ≠ a
  measurement; recording it as observed would poison the fit exactly as fabricated entropy neutrals did.
• **Weight earned, not assumed** — `w_effort` pinned to 0 until `_MIN_EFFORT_SAMPLES`(30) observed samples with
  BOTH outcome classes, same gate as w_entropy. So live behaviour is unchanged today and engages on evidence.
• **Renormalisation generalised to 3 components** — only observed components contribute, divided by their
  weights. With entropy absent and effort present, competence+effort renormalise to 1 instead of leaving 40% of
  the mass on a neutral stand-in. `_composite_for` mirrors `score()` exactly (the fit must optimise the formula
  the agent actually evaluates).

LIVE-VERIFIED: a multi-tool turn recorded `"effort_component": 0.625, "effort_observed": true, "domain": "fs"`.
learning_health now reads **`features: 1/4 live (effort_component)`** — separation −0.25 — up from 0/4.
(Sign is negative because the probe turns so far were successful long turns; it will settle as samples accrue.)
Tests: test_confidence_turn_effort.py (17). FULL SUITE 9477 passed / 0 failed. Docs: core/confidence.html.
DEPLOYED (42775→43167, health ok).
WATCH: once ≥30 effort-observed samples of both classes exist (~1 day at ~60 samples/day), the idle refit should
raise `w_effort` off 0 and the base-rate comparison line should flip from "LOSES TO" to "beats". If it does not,
the feature is not transferring from the trajectory corpus to live turns and should be re-measured, not re-tuned.

### 2026-07-27 (later 4) — Self-learning review: the confidence score has NO live input (measured)

Operator: "any other improvements to the self-learning subsystem?" → "proceed with everything."
I measured instead of guessing, and the headline **corrects a claim I made earlier in the same session**.

**⚠️ SELF-CORRECTION — my first read was wrong.** I reported AUC 0.679 and "Platt recalibration beats
the baseline by 18%". That was computed on the STORED `composite` field, which mixes two scoring regimes
(early records were scored with the pre-fit default w_e=0.5). Recomputed LEAK-FREE from the stored components,
**AUC = 0.473 — below chance**. The composite has no discrimination. Lesson: recompute a metric from raw
components before trusting it; a stored score reflects whatever formula was live when it was written.

**THE DIAGNOSIS: 0 of 3 confidence inputs are alive.** Measured over 1208 samples (varies? separates?):
| feature | distinct | separation (succ−fail) | verdict |
|---|---|---|---|
| entropy_component | 2 | +0.0002 | DEAD (upstream refuses logprobs on tools+stream) |
| competence_component | 270 | **−0.0008** | DEAD — varies plenty, predicts NOTHING |
| uncertainty_pressure | 1 | 0.0 | DEAD (tracker logged 2 records in 20 days) |
So the binding constraint is **feature quality — not label supply and not calibration**. Competence is a
per-DOMAIN historical average: near-constant within a domain, therefore structurally unable to discriminate
individual turns. A useful confidence score needs an input that varies per TURN. Deliberately NOT invented one
here — that is a design decision for the operator, and inventing features to make a number move is exactly the
verification-theater this project fights.

SHIPPED:
• **Probability recalibration stage** (Platt `sigmoid(a·c+b)` after the weight fit; raw + base-rate Briers
  persisted so the comparison can never be lost). On live data it is **REJECTED by design** — the optimum is
  `a≈−0.08`, a NEGATIVE slope that would invert the agent's confidence ordering, and near-flat enough to
  collapse every turn onto the base rate leaving `below_threshold` inert. Guard: reject any slope < 0.5 and
  WARN. Verified it engages correctly on an informative-but-compressed corpus (adopted, beats base rate).
  Newton/IRLS not gradient descent: GD was still at a=1.28 after 4000 steps and only reached a≈−0.08 after
  ~60000 — opposite sides of the guard, so an under-converged run would have adopted a map the converged fit
  rejects. The answer must not depend on the iteration budget (IRLS: <20 iters, 0.04s).
  UNWEIGHTED after trying class-balancing: balancing optimises a reweighted distribution, so the probabilities
  describe a 50/50 world that doesn't exist and unweighted Brier got WORSE. Imbalance belongs to the threshold
  search (Youden's J), which is prevalence-independent.
• **`domain` was never recorded** (BUG): computed for the competence lookup, then dropped — all 1208 samples
  carried `domain=""`, making per-domain reliability impossible. Now recorded (live-verified `"domain": "fs"`).
  Hoisted the derivation so it's defined on the streamed path too (it would have been a NameError there).
• **User corrections now tick the LESSON arm negative**, not just calibration. A correction is the strongest
  FAILED signal available, and those turns are exactly what the outcome-gated arm exists for: the turn looked
  clean, so it was stashed "awaiting a late verdict", no verdict ever landed, and the stash entry was evicted
  UNCOUNTED. Reuses the existing flush helper (no-op when the turn recorded inline).
• **Correction negatives carry `entropy_observed`** — an observed negative is the single most valuable sample
  type (negatives ~4%, observed ~0%); the stash was silently downgrading it. (Gap I introduced in later-3.)
• **Feature-health telemetry**: per-feature distinct/separation/dead, `features: 0/3 live — NONE discriminate`.
  Death test is SEPARATION, not distinctness — a 2-valued feature that splits cleanly is useful, while
  competence has 270 values and separates by −0.0008.

INVESTIGATED, NOT A BUG: the outcome-gated arm sees only ~5% of retrievals (2288 retrievals → 120 ticks). That
is verdict-gating working as designed — a clean turn with no verdict is deliberately evicted UNCOUNTED so the
arms never fill from mere absence-of-failure. Loosening it would manufacture unverified successes. Note the two
arms use DIFFERENT success definitions (calibration books clean turns as success → 96% positives; the lesson arm
refuses to) — the lesson arm's conservatism is the better design.
Also: negative supply is already mined from every available source (execution failures, verifier refutes,
budget exhaustion, user corrections); 4% is the agent's real recorded failure rate, not a collection gap.

Tests: test_calibration_probability_map.py (31). Two of my own new tests caught real never-raises violations in
the telemetry helpers (corrupt `frequency`, non-numeric `outcome`) — both fixed.
FULL SUITE 9460 passed / 0 failed. Docs: core/calibration.html. DEPLOYED (39274→41264, health ok).
NEXT (operator decision): give confidence a per-TURN feature — candidates already computed at scoring time are
tool-count, whether a repair/steer fired, and context pressure. Until then the honest state is "no signal", and
the telemetry now says so out loud.

### 2026-07-27 (later 3) — The five measured improvements: entropy unfittable, PRM unread, graduation impossible, episodes blind

Operator: "fix the calibration entropy, drop PG-manual re-ingest (done), then proceed with the rest of your
suggestions, implement them all." All five items from the later-2 measured eval. THEME: four of the five were
**instruments or loops that could not possibly work**, and in three cases the metric that should have shown it
was itself broken — measure the mechanism, not the summary number.

**#1 CALIBRATION ENTROPY — structurally unfittable, now fixed.** Live: 1200/1201 samples had
`entropy_component` EXACTLY 0.5, fitted `w_entropy` 0.0. ROOT CAUSE found by instrumenting the running agent
(temporary ENTPROBE log, since removed): `final=False tools_in_payload=True → request_logprobs=False`.
llama-server 400s on `logprobs`+`tools`+`stream` (re-confirmed live), and the loop attaches tools on every
NON-final generation — and a normal turn answers on the first pass, so `is_final_generation` is almost never
true. The logprob-bearing generation essentially never happens. Those turns then recorded the neutral 0.5
STAND-IN as if it were a measurement → zero variance → any w_e>0 could only drag composites toward 0.5 → the
grid was GUARANTEED to pick 0. Not a tuning problem: the data made the parameter unlearnable.
FIX (3 parts): (a) `entropy_observed` on ConfidenceReading + CalibrationSample (legacy records default False —
truthful, they WERE fabricated); (b) **missing-feature renormalisation** — an unobserved sample scores on
competence ALONE, via `_composite_for` which mirrors `score()` exactly so fit and scorer optimise the same
formula. This is load-bearing: my first attempt (fit w_e on the observed subset, apply to all) drove Brier
0.075 → 0.219 because unobserved samples got blended with the stand-in. Caught it by testing 4 regimes before
committing; (c) w_entropy stays pinned at 0 below `_MIN_ENTROPY_SAMPLES`(30) observed samples of both classes,
with the reason logged + `n_entropy_observed` persisted. Also widened the request gate to `"tools" not in
payload` (dropping the redundant `is_final_generation` term — strictly narrower than the server constraint).
VERIFIED across 4 regimes: all-neutral → w_e 0 (no regression); observed+informative → **w_e 1.0 (previously
impossible)**; observed-but-noise → w_e 0 (declines noise); mixture → w_e 1.0 @ Brier 0.093 (vs 0.219 naive).
LIVE: new samples carry `entropy_observed: false` and `composite == competence_component` exactly.

**#2 PRM — trained every idle cycle, read by NOTHING.** Both consumers off: `.score()` → MCTS turn-start
(module-gated, no flag can enable) and `.uncertainty()` → frontier self-play (`--frontier-selfplay`, default
False since 07-09 AND absent from the live exec line — I checked the launcher, not the docs). 41 `prm_train`
events in one ledger window writing a checkpoint nothing read, while logging "value model refit" — reads like
learning progress. Phase 2.7 now SKIPS with an explicit reason when no consumer is live. Deliberately a RUNTIME
check, not a deletion: flip either consumer and training resumes next idle pass, no code change.
NOTE the earlier telemetry named only `.score()` — it understated how dead this was; now reports both.

**#3 LESSON GRADUATION — could never fire.** Measured against the live 50-lesson playbook: 3 lessons met
`frequency>=5`, 17 met the code gate, **ZERO met both** → no candidate, ever. Both gates wrong in opposite
directions: (a) freq>=5 unreachable (38/50 sat at 1) and the 3 that reached it are behavioural guidance;
(b) the code detector was a substring scan including `"with "` and `"return "` — ORDINARY ENGLISH. "joining all
results *with* the exact delimiter" registered as code (17/50 false positives). Structural detection finds 2.
FIX: reusability = `freq>=3 OR verified` (a verifier-confirmed lesson is at least as trustworthy as one seen
5×); `_looks_mechanizable` requires real syntax; candidates sorted by strength so the 2/cycle budget goes to the
best. HONEST OUTCOME: 1 eligible, and that is CORRECT — this playbook is mostly behavioural heuristics that
should never become Python tools. Tuned to admit the genuinely mechanizable, NOT to force graduations; TDD gate
still the backstop. learning_health now reports `eligible (reusable ∩ mechanizable)` so "0 graduated" is
explainable. (My own new test caught a real defect while writing it: `_graduation_eligibility` crashed on a
corrupt `frequency`, violating the module's never-raises contract.)

**#4 EPISODE CONTEXT/CLUSTER BACKFILL — 5.2% → 96.6%.** 165 pre-07-26 episodes had empty `context`/`cluster_id`,
so `get_episodes_by_cluster` and `search_recoveries` (greps context for FAILED markers) were blind to ~95% of
the corpus. The ACTIONS were always persisted to `episode_actions`, so both fields were reconstructed by
REPLAYING the same derivation the live path uses — not guessed. `scripts/backfill_episode_context.py`: dry-run
default, idempotent, and REFUSES to invent data for the 6 action-less episodes rather than writing placeholders.
Applied after a DB backup (`episodic_memory.db.pre_context_backfill.bak`): 159 updated, 168/174 (96.6%).
Cluster retrieval went from ~9 total to fs=41 / memory=40 / fetch=22 / shell=19; `search_recoveries` now returns
real recoveries.

**#5 IDLE-LOOP "IMBALANCE" — WAS A MEASUREMENT ARTIFACT; no rebalance made.** My later-2 note flagged
dream 231 vs reflection 21 as a starved loop. WRONG: reflection only writes a ledger event `if report.outcomes`,
AND deliberately skips ticks whose trajectory corpus is unchanged since an all-duplicate pass (a correct 07-18
fix); dream records every cycle. I verified the skip gate re-arms properly (arms only on a do-nothing pass,
clears when it reflected or errored). So the counts are per-phase event records with different recording
policies, NOT a workload budget — the same artifact class as the FAILURE-arm "inert" false alarm. Changed the
telemetry to say so rather than "fixing" a non-problem. The one GENUINE idle waste (PRM, #2) is fixed.

**BACKLOG CORRECTION CONFIRMED:** PG-manual re-ingest formally dropped (later-2 measured the corpus healthy —
1.2% "fusion" hits were all legitimate PostgreSQL C-API camelCase; docs excluded from recall at 4 sites).

Tests: test_calibration_entropy_observed.py (15) + test_lesson_graduation_gates.py (19) +
test_episode_context_backfill.py (11) + 3 PRM wire-or-retire tests. Two pre-existing tests updated to the
corrected behaviour (the logprobs gate pin — the no-tools check IS the safety property, the
`is_final_generation` term was redundant; and the PRM phase fixtures now grant a live consumer).
FULL SUITE 9429 passed / 0 failed. Docs: calibration/confidence/dream/episodes/introspect.
DEPLOYED (listener 37161→39274, health ok); functional_live_test 32/32.
WATCH: `entropy observed on N/1205` should start climbing only if tool-free generations occur; if it stays at 0
the honest conclusion is that this upstream+tool-mode combination cannot supply token entropy at all, and the
entropy term should be retired rather than carried as a permanently-pinned zero.

### 2026-07-27 (later 2) — Dead-code deletion (qwen variant) + honest cognitive-layer telemetry

Operator: "delete qwen_bridge and fix mcts/arbiter in the logs." Both were the DEFERRED items from the
same-day hunt (see later-1).

DELETED (verified unreachable first): `core/agent_qwen.py` + `tools/qwen_bridge.py`. Nothing in src/, scripts/,
bin/ or interface/ ever imported or constructed `GhostQwenAgent` — the only inbound references were tests.
Removing the pair also retires two latent defects that would have bitten the first real user of the variant
(`_run_coro_blocking` ran each coroutine on a FRESH loop while blocking the caller's loop in `join()` → whole-agent
freeze + cross-loop errors; system-injected kwargs not excluded from the model-param pass-through → duplicate-kwarg
TypeError on a hallucinated `sandbox_dir`). `qwen-agent` dropped from requirements.txt with it.
KEPT deliberately: `soundfile` (its requirements comment claimed it was a qwen-agent transitive dep, but
`interface/externals/tts_stt/voice_server.py` imports it — comment retargeted, pin stays) and
`test_agent_qwen_syntax_healer` (despite the name it tests the MAIN agent's Qwen-DIALECT `<tool_call>` healer,
which is very much live — nothing to do with the deleted variant).
Test surgery: deleted `test_qwen_bridge_import_error.py`; excised the qwen sections from `test_deep_audit_fixes.py`,
`test_tools_audit_fixes.py` (Fix #10) and `test_open_audit_fixes.py` (CRITICAL-2), leaving their other tests intact;
docstring corrected in `test_bughunt_unit18_agent.py`.

TELEMETRY HONESTY (the "dead consumer announced as live" class):
• main.py boot: "MCTS + Hypothesis testing enabled (opt-in via --deep-reason)" → now reports the EFFECTIVE state
  per module constant: "hypothesis testing ENABLED · MCTS turn-start hint OFF (module toggle — attached but never
  invoked)". Hypothesis grounding is the part that actually runs (System-3 pivot); MCTS is hard-gated by
  `_MCTS_TURNSTART_ENABLED = False`, which no flag can flip.
• main.py "PRM ↔ MCTS … LLM simulation bypassed" and agent.py's idle-retrain " · MCTS now scoring via PRM" both
  announced an upgrade to that never-invoked consumer — the agent.py string also landed in the OPERATOR-FACING
  activity ledger. The wiring is kept (free, correct the moment the gate flips); only the CLAIMS are now gated.
• Arbiter: the metacog BOOT log already reported `arbiter="off (module toggle)"` honestly (earlier fix), so nothing
  to correct there. I tried also SKIPPING arbiter construction when the gate is off — REVERTED: it broke 32 tests
  that exercise arbiter mechanics directly, and the construction is only a pair of closures over the ALREADY-loaded
  embedder (no model load, no network), so there was no waste to reclaim. Documented in-place so the next reader
  doesn't re-litigate it.

Docs: `tools/qwen_bridge.html` + `core/agent_qwen.html` replaced with TOMBSTONE pages (what was here / why removed /
what replaced it) so existing links don't 404; nav entries stripped from 15 pages; installation.html dependency row
rewritten (qwen-agent gone, soundfile retargeted to the voice server); architecture.html tree line corrected.
Docs link-check: 3 broken links remain repo-wide, all PRE-EXISTING and unrelated (`core/registry.html`,
`distill/collector.html`, `workspace/activity.html` — targets that never existed).
FULL SUITE: 9381 passed / 0 failed / 13 skipped. DEPLOYED (listener 33467→35513, health ok); new boot line confirmed
in the live log, and the false "PRM ↔ MCTS" line is absent from that boot.

IMPROVEMENT EVAL — MEASURED, not estimated (operator asked "which are most impactful"). Two BACKLOG CORRECTIONS
came out of measuring instead of trusting the catalogue:
• ⚠️ **"PG-manual re-ingest (96.7% of vector store, 100% corrupt)" — ranked #3 in the 07-26 eval — IS A
  NON-ISSUE. DROP IT.** Measured the live store: 8438 embeddings, 8280 (98.1%) from `postgresql-19-A4.pdf`.
  Sampled + scanned for the splitter's word-fusion signature: only 96 chunks (1.2%) match, and every one is a
  legitimate PostgreSQL C-API camelCase identifier (`paramTypes`, `grantMask`, `forRead`) — NOT fusion. Avg chunk
  1157 chars, prose reads clean. The corpus is HEALTHY. It also does not pollute recall: the ambient paths filter
  `{"type": {"$ne": "document"}}` at 4 sites in vector.py (docs reach the model only via the doc-scoped
  `knowledge_base(action='query')`). Nothing to re-ingest; the missing source PDF no longer blocks anything.
• ❌ **The 07-26 WATCH ITEM RESOLVED NEGATIVE: streamed calibration entropy still never lands.** The fix note said
  "verify the streamed calibration samples now carry real (non-0.5) entropy and w_entropy actually moves." Measured
  `calibration.jsonl` (1201 samples, 07-07 → 07-27): **1200 of 1201 have `entropy_component` exactly 0.5** (the
  neutral default) — ONE sample in three weeks carries a real value (0.7253). Hence `entropy distinct values: 2
  (DEGENERATE)` and the fitted weights `entropy 0.0 / competence 1.0`. The confidence composite is running on
  competence ALONE; the entropy half is structurally dead, not merely under-weighted. This is the #1 open item.
Other measured signals (from the now-working learning-health telemetry): PRM retrains on every idle cycle (41
`prm_train` events in the recent ledger window) while its ONLY consumer is gated off — `.score()` OFF because MCTS
turn-start is disabled — so it is pure idle-time cost producing a checkpoint nothing reads (wire-or-retire, §3
names the intended grounded replacement); 0 of 50 lessons GRADUATED (9 verified, 2 stale/prune candidates,
mean hit-rate 0.611); episode context/cluster coverage 9/174 (5.2%) because the 07-26 field-population fix only
applies to NEW episodes — the 165 older ones need a one-shot backfill for cluster retrieval to see them; idle-loop
budget is lopsided (dream 231 vs self_play 133 vs reflection 21) and reflection is the arm that mints lessons.

### 2026-07-27 (later) — Least-audited-systems bug hunt: the SQL guard that never ran + two reports that went dark

Operator: "locate systems that we haven't swept for bugs and improvements, analyze them and fix/improve."
Journal-driven target selection: every deep review since 07-20 hit the same core stacks (memory, LLM/routing,
sandbox, metacog, turn-loop, verifier, project-autonomy), so the five clusters chosen were the ones with the
LEAST attention since the generic 07-03/04 sweep — and two of them (`core/sessions.py`, `tools/composed_skills.py`)
shipped 2026-07-11, AFTER that sweep, so they had NEVER been audited. Five parallel read-only agents:
tasks/scheduling · media+bridge tools · database+games · sessions/interface/composed-skills · deep-reason+introspect.
Agents were pre-loaded with the §4B known-deferred items so they didn't re-report them. Every HIGH re-verified at
source by the orchestrator (several were reproduced against LIVE data/DB before fixing).

CROSS-CUTTING THEME: **a guard/report that was never actually running.** Four of the five HIGHs are not logic
errors — they are subsystems that LOOK wired and report success while doing nothing. Two were invisible because
their tests build fixtures too small to cross the threshold that breaks them.

FIXED (all verified at source; HIGH first):
• **SQL destructive-statement guard was INOPERATIVE in prod** (validators.py). `validate_sql` delegated
  multi-statement splitting to `sqlparse` — never in requirements.txt, NOT installed — so the `ImportError`
  fallback (all `^`-anchored regexes) was the only path that ever ran. Live-verified clean-validating:
  `SELECT 1; DROP TABLE web_order_line_options;`, `WITH d AS (DELETE FROM t RETURNING *) …`, `DO $$ … EXECUTE
  'DROP TABLE t' … $$`, `/* c */ DROP TABLE t`. psycopg2 on the autocommit connection runs the whole batch, and
  the default DB holds real tables → live data loss, no confirm, no guard message. REWRITTEN self-contained:
  `_mask_sql` blanks literals/`--`/nested `/* */`/dollar-quoted bodies (length-preserving), splitting + paren
  balance + verb detection all run on the mask, verbs match at STATEMENT level (not position 0) with the WHERE
  test scoped to the match's own paren group, dollar-quoted bodies require `confirm`. No new dependency.
• **Vision could OOM the host from a KB-sized PDF** (vision.py). The 50 MB *file* cap cannot bound `get_pixmap`;
  a 788-byte PDF declaring a 10000×10000pt page → 1.2 GB pixmap at the fixed 2× zoom, ×10 pages, on the box that
  runs the LLM pinned in RAM. Zoom now derived per page from `_MAX_PDF_PAGE_PIXELS` (A4 keeps full 2×).
• **Cron tasks fired in LOCAL time** (tasks.py). A pre-built trigger instance never inherits the scheduler's
  `timezone="UTC"` — `from_crontab(expr)` defaulted to `get_localzone()` (verified: Europe/Athens), while
  registry.py tells the model "cron times are UTC — convert first". Every cron task fired 3h early and drifted
  with DST. Also: all three add_job sites inherited APScheduler's 1s misfire grace, and the loop stalls >1s
  routinely → fires SKIPPED silently, nothing in the ledger. Now `timezone="UTC"` + `misfire_grace_time=300` +
  `coalesce=True`; `list` converts next_run_time to UTC explicitly.
• **`introspect action='activity'` blind since ~2026-07-13** (introspect.py). `log.read_since(0)` inherits
  `limit=200` → it read the 200 OLDEST lines of a never-rotating ledger. Reproduced against the live ledger
  (1518 records, newest that day): returned records ending 07-13, so the report said "No background activity"
  even at the 336h max window — dead from exactly the point the 07-17 chat-noise decision made it the ONLY home
  of routine-maintenance records. Tests passed because fixtures are <200 lines. Now reads the TAIL.
• **Fat-client session replay compounded the conversation** (sessions.py). `merge_history` required an EXACT
  stored-prefix match; any divergence fell through to `stored + incoming`, re-appending the whole conversation
  every turn (measured 5→11→19→29, quadratic) until the 400-cap filled with duplicates. Two permanent triggers:
  an aborted stream (server persists the full reply before streaming, client keeps a partial one) and the cap
  itself (stored loses its oldest → client replay can never prefix-match again). Now tolerates divergence at any
  offset — the replay is authoritative and REPLACES stored; thin clients unaffected.
• Learning-health background section never rendered (learning_health.py): `_activity_counts` keyed on
  kind/type/category but the record serialises `phase` → always `{}` against 1500+ real records, in the very
  instrument built for the keep/kill watch. Now keyed on `phase` + honest recent-window filter.
• Postgres pool/session (database.py): pool keyed on the RAW URI (`postgres://` vs `postgresql://`, added query
  params → a permanent extra connection each, walking toward max_connections) → now keyed on the resolved
  endpoint + bounded with close-on-evict; only `statement_timeout` was reset between calls, so a `SET search_path`
  silently changed which tables later unqualified queries resolved to (and an uncommitted batch left the pooled
  conn idle-in-transaction holding locks) → now `rollback()` + `DISCARD ALL` per action; the 200k-char cap ran
  AFTER fetch+tabulate each built a full copy (300 × multi-MB cells ≈ GB resident → OOM not truncation) → cells
  clipped at 4k before render.
• report_pdf: ATX headers matched INSIDE fenced code blocks → `# install deps` in a ```bash block became an h2
  and the fence markers were stranded across two sections (very common in this agent's own reports); splitter now
  tracks ```/~~~ fences. Also moved `_sections_from_files` / `_available_files_hint` / `_build_html` off the event
  loop (25 MB/file reads + full `sorted(rglob("*"))` + 200k-char markdown parse were inline), short-circuit once
  the report cap is blown, and `_md_to_html` degrades on ANY exception (caught only ImportError).
• Games: tic-tac-toe bricked after one move — `load` accepted a side-to-move contradicting the parity invariant
  it enforced two lines earlier, so "agent opens as O" produced a state the endpoint itself then 422'd forever.
  Either side may now open (|nx-no| ≤ 1) and the turn is FORCED when counts differ (closes the double-move hole).
  Blank `state=""` silently started a new game (chess 422'd — the two disagreed) → 422. `extract_labeled` was
  first-wins while `extract_move_text` is last-wins → with the live THINKING model the played move and its stated
  reasoning came from different candidates; now last-wins. NOTE: games auth was NOT forgotten in the 07-13
  rollout (router-level `Security(verify_api_key)`, mounted before the catch-all) — verified, clean.
• Triggers/metacog: `RepetitionCounter.observe(fname)` keyed on tool NAME and counted SUCCESSES → three reads of
  DIFFERENT files tripped LoopDetected, and each trip burned one of the executing task's MAX_REVISIONS via the
  ReplanBridge (after 3 benign trips a REAL failure could no longer revise). Now keyed tool+args-hash.
  `ReplanBridge._revisions` was an unbounded list fed by every event incl. the 300s degraded-telemetry heartbeat
  → `deque(maxlen=256)`.
• Interface: upstream chat failure was recorded in `task["error"]` but never EMITTED → agent down = HTTP 200 +
  empty SSE + bare "No response", no diagnostic (live stream and resume both); now an `event: error` frame, and
  resume replays terminal truncation/error too. `GHOST_API_KEY=""` passed the presence check but can never match
  `compare_digest` → every route incl. `/` 401s while launchd KeepAlive keeps the dead service "up" (and the
  agent treats "" as auth-OFF, so the two diverged) → now raises at import. `stream_generator` null-checks its
  task (janitor can pop it before the first iteration → TypeError instead of the intended marker).
• Composed skills: a step whose tool RETURNED an error string (the codebase norm — no `error` key) rendered as
  "FAILED — unknown error", so the model had zero diagnostic to recover from; now falls back to the result body.
  `_load`/`approve` skipped name validation and the live registry still holds a dotted legacy entry
  (`auto.generic.…`) → approving it would emit a dotted LLM function name; now validated on both paths.
  `save()`'s mkdir moved inside its try (raised through record_usage → discarded a successful macro's results).
• agent.py hypothesis adjudication: survivor indices parsed with bare `int(_i)`, but the prompt renders
  hypotheses as "H0:/H1:" and models reply with labels → ValueError → the silent `except: pass` skipped the whole
  promote/demote block, reinstating the "survivors = the hypotheses the evidence did NOT confirm" bug the
  surrounding comment claims was fixed. Now tolerant per-item coercion.
• main.py watch runner: the edge was persisted (`set_watch_state(True)`) BEFORE dispatch, and the inner defer
  check then discarded the reaction FOREVER while the condition stayed true (next tick sees no edge) — same loss
  on a kill mid-reaction, i.e. every deploy. Edge is now consumed only on a real dispatch (`_run_proactive_task`
  returns dispatched/deferred). A deferred CRON job is nudged to retry in 60s rather than losing the whole period
  ("will retry next tick" was only ever true for interval jobs).

Tests: tests/test_bughunt_fixes_2026_07_27.py (64 new). Two pre-existing tests updated to match corrected
behaviour (both asserted the OLD literal shape, not the intent): test_recent_fixes' DB-timeout test (DISCARD ALL
now precedes the SET) and the report_pdf hint-wiring source assertion (now called via to_thread).
FULL SUITE: 9389 passed / 0 failed / 13 skipped. Docs: 11 pages (audit_fixes + database/vision/report_pdf/tasks/
introspect/composed_skills/sessions/triggers/game_routes/web_server).

VERIFIED-CLEAN (worth NOT re-auditing): chess rules delegate correctly to python-chess (castling/en-passant/
promotion/repetition/fifty-move all covered); games are stateless per request (no shared-state race, no growth);
`table_name`/`timeout_ms` injection closed; DB host-redirect guard resolves libpq query-param overrides incl.
keyword/value DSNs; connection lifecycle closes on every exception path; vision SSRF hop-revalidation + streamed
byte caps hold; image_gen error paths (empty-b64, content-filter, data-URI, coercions) all sound; composed-skill
recursion blocked three ways + dataflow validation correct; session persistence atomic with corrupt-file
tolerance; interface auth covers every proxy route (only /static and /sw.js open); JobRegistry lock discipline
and task-ref retention correct; TriggerBus subscriber isolation + capped history.

DEFERRED (logged, lower value): `qwen_bridge` kwarg-collision + `_run_coro_blocking` cross-loop hazard (the
GhostQwenAgent variant has ZERO instantiation sites — best closed by deleting the variant, as §4B already notes);
MCTS + arbiter are dead-by-design (hardcoded `_MCTS_TURNSTART_ENABLED=False` / `_METACOG_ARBITER_ENABLED=False`)
yet the boot log still prints "MCTS + Hypothesis testing enabled" and the PRM idle-retrain keeps wiring scorers
into the never-invoked reasoner — misleading operator telemetry only, worth gating the log lines; tic-tac-toe
double-winner boards still accepted; `_available_files_hint`/activity-ledger rotation; per-cell clip means a
genuinely huge single cell is now summarised rather than partially shown.

### 2026-07-27 — Overnight-log eval → idle-loop yield fixes (counterfactual gate, mastery floor, curriculum balance, calibration entropy)

Operator: "read the agents log that run overnight and evaluate" → "proceed with all fixes." The overnight run
(00:52–08:11) was CLEAN on stability — 0 tracebacks, 8/8 self-play sims SUCCESS — but near-zero yield: 2 new
lessons, 0 graduated, 6/9 idle cycles spent replaying already-solved challenges, one ~2h slot forfeited to a 3/3
generation-rejection streak. Five fixes, all DEPLOYED same day:
1) **Counterfactual learning-state gate** (`core/counterfactual.py`): 45/45 lifetime replays were
   stable-pass/generalized, 0 regressions — the arm re-measured an UNCHANGED learning state on ~40% of the idle
   budget. `should_replay()` fingerprints skills_playbook+auto_skills (SHA-1) and skips the batch when unchanged
   since the last DECISIVE batch (inconclusive batches don't stamp; stamp is post-batch so replay-written lessons
   re-arm honestly); the slot falls through to fresh self-play. Kill: GHOST_COUNTERFACTUAL_GATE=0.
2) **Mastery floor waiver** (`memory/frontier.py`): `description_length` is a tool-invocation count with a
   structural floor (~4), so C7's `any(delta>0.05)` was arithmetically unreachable at best_length=4 — concurrency
   6/6 first-try, sql 6/6, regex_parse 5/5 pinned unmastered forever and endlessly re-picked. New
   MASTERY_LENGTH_FLOOR=5: at/below it the 5-run first-try streak alone decides. Above the floor C7 unchanged.
3) **Curriculum balance**: 82% of 154 runs sat in data_analysis/python_general (web_automation: 1 lifetime run).
   `cluster_run_weights` (1/(1+runs)) now biases BOTH random-template draws
   (`pick_random_template(cluster_weights=…)`), and `least_practiced_clusters` feeds a COVERAGE TARGET block into
   the LLM generation prompt when the seed doesn't pin a cluster.
4) **Generation-slot resilience** (`core/dream.py`): diversity window forward-feed widened 5→12 heads (the
   overnight 0.91/0.93-overlap dupes sat just outside the 5-slice); a 3/3 quality-gate rejection now falls back
   to a deterministic template (`_tpl_source="rejection_fallback"`) instead of forfeiting the slot.
5) **Calibration entropy de-degenerated** (`core/agent.py`): 1179/1180 samples carried entropy_component=0.5 —
   the internal upstream stream (EVERY sim/CLI turn) never requested logprobs and the finalize fallback hardcoded
   0.5, so w_entropy was unfittable. Logprobs opt-in hoisted to the internal path, chunks observed into an
   EntropyTracker (live MTP upstream emits logprobs ~1 chunk in 7 — verified by direct probe; sparse is fine),
   reading stashed req-id-tagged (`_entropy_norm_pending`) and consumed by `_record_calibration_safe`. GOTCHA
   caught on the post-deploy probe, not by the 9324-test suite: llama-server 400s `logprobs` on tools+stream
   payloads — the opt-in is scoped to FINAL generations (which never carry tools) + a `"tools" not in payload`
   belt-and-braces check. ROOT CAUSE of the darkness, established by live probes: with --native-tools ON
   (default since 2026-07-17) every tool-attached turn is logprob-blind BY SERVER CONSTRAINT, and most
   interactive answers land ON a tools-attached turn (model answers directly instead of transitioning to a
   forced final generation — verified on 3 live probes incl. a tool-work turn). The lone real-entropy sample
   ever is dated 2026-07-18 — the native-tools OFF/ON toggle window. So entropy coverage is now: forced final
   generations, truncation continuations, SSE final streams — real but SPARSE. DECISION RULE for the watch:
   if learning-health `entropy_distinct_values` is still <3 after ~2 weeks (~2026-08-10), retire w_entropy
   (it is already 0-weighted, so no behavioral change either way); the alternative (stripping `tools` from
   predicted-easy turn-1 payloads via the complexity router) is a routing change — decide separately.
Also: **self_consistency RETIRED** (module deleted — learning-health had it INERT, grep confirmed zero callers
incl. offline tooling; `optim.trainset._dedupe_self_consistency` stays for old batch_id corpora); **FAILURE-arm
inertness warning rewritten** in learning_health — the old fail-ONLY-lesson test was a metric artifact at a ~96%
pass rate (live: 29 failed ticks already flowing → arm was never inert; warning now requires ZERO failed ticks
against ≥20 succeeded); launcher comment fixed (bin/start-ghost-agent.sh said frontier-selfplay "default: ON",
actual default OFF since #27b 2026-07-09). NOT changed (watch items, not defects): slack-bot 8MB unrotated .err
(httpx INFO noise, no leak — one poller per process confirmed) **[RESOLVED 2026-08-01: log-hygiene rework — see
§6 "Slack notification workflow review"; .err truncated, httpx silenced, self-rotating .info.log]**; native
tool_call repair fired in 4/8 overnight
sessions (guards worked; known upstream issue). Tests: test_counterfactual.py::TestReplayGate,
test_selfplay_curriculum_2026_07_27.py, test_calibration_entropy_wiring_2026_07_27.py, test_learning_health.py
(rewritten arm tests). Docs: memory/frontier.html, core/dream.html, core/calibration.html, core/entropy.html,
core/challenge_templates.html.

### 2026-07-26 (later 10) — Web-search turns falsely REFUTED: the judge is a 4B model, not a bad prompt

Operator: "web searches cause the verifier to refute … how can we fix this elegantly?" (cases: B3 "latest postgresql
version" LATE REFUTED 90%; req 7779fe10 PDF-ingest LATE REFUTED 100% for stats the tool literally printed).
DIAGNOSIS (measured, not guessed): built a 3-case A/B repro from the REAL trajectory (entailed superlative /
fabricated version control / unit-conversion) and ran it against BOTH judges. Worker route = nova = **Gemma 4 E4B
(4B)** scored 1/3 — refuted the correct PG answer, and asserted "49152 bytes is not exactly 48 KB" (wrong
arithmetic). Main = Qwen3.6-35B scored **3/3**, incl. correctly refuting the fabrication. The prompt was never the
root cause — I iterated it first and got only partial improvement, which is what prompted measuring the model.
FIX (screen cheap, confirm expensive — the gate discipline already used elsewhere here): the 4B still screens
every turn (no added latency / no main-slot contention on the common path), but a REFUTED verdict is
re-adjudicated on the MAIN model before it can act — `_escalate_refute` + `force_main` routing that bypasses the
critic pool AND the worker route. Strong model wins on disagreement; ANY error keeps the original verdict, so
escalation can only reduce false refutes, never reduce availability. Cost: one main call on the rare refute path.
Why it matters: REFUTED scrubs the turn's lessons, files follow-ups, shows the user a correction banner and marks
the trajectory FAILED (corpus poisoning for every downstream learner) — and all of those are gated on REFUTED
specifically, so an overturn to CONFIRMED *or* UNCERTAIN is safe. Also hardened both adjudication prompts with a
DERIVED-facts rule (arithmetic/unit-conversion, ordering/superlatives, evidence-marked classification "19 is Beta
⇒ newest stable is 18.4", counts) + "you cannot know today's date, so 'not verifiable as current' is never
grounds for REFUTED". LIVE A/B AFTER: PG REFUTED(0.9)→UNCERTAIN, unit-conversion REFUTED→CONFIRMED(1.0),
fabrication control still REFUTED(0.95). Kill switch GHOST_VERIFY_ESCALATE_REFUTE=0. Tests:
test_verifier_refute_escalation.py (8, deterministic). Doc: core/verifier.html. NOTE for later: the standing
`--critic-nodes` slot is the permanent home for this if a mid-size judge ever lands on a spare box.

### 2026-07-26 (later 9) — Episode reconcile made dedup-aware (the "N of M missing twin" that never dropped)

Operator asked about the boot line "Episode vector reconcile: 25 of 167 episodes have no vector twin —
re-ingesting." Read-only classification of the live stores: 167 episodes, 142 have a twin, 25 missing — and
ALL 25 are DEDUP-COVERED (share their trigger text with an episode that DOES have a twin), 0 genuinely invisible.
Root cause: the vector store dedups on document text (md5 of trigger), so same-trigger episodes share ONE entry;
the others have no own-id twin but ARE reachable via the shared entry. The old reconcile counted them as
"missing" and re-add()'d every boot (each a no-op) → number never dropped, ticked up over time, false alarm on
the operator's stream. FIX: reconcile_vector_index now reads include=["metadatas","documents"] and compares each
episode's would-be document against the already-indexed documents — present → dedup-covered (skip, INFO log);
absent → genuine hole (re-ingest). Live effect: 0 genuine re-ingests, 25 correctly recognized as dedup-covered,
zero wasted work. Tests: test_episodes_recall_fixes.py (+2 dedup cases; FakeCollection.get now returns documents).
Doc: memory/episodes.html. Deploy pending full-suite.

### 2026-07-26 (later 8) — Fresh-log eval: fixes confirmed live + workspace_track read-intent steer

Re-read ghost-agent.log post-deploy. GOOD: boot clean; the later-3 EPISODE BOOT RECONCILE fired in prod ("24 of
165 episodes have no vector twin — re-ingesting" → "re-embedded 24"); metacog calib loaded threshold=0.71
w_entropy=0.00 (confirms the degenerate-entropy state the new telemetry flags — will diversify as streamed
real-entropy samples accrue). ACTIONABLE: live req 30 ("what's new today?") FAILED — the model called
workspace_track (the WRITE tool) twice with NO action, wanting to READ recent events, and burned 2 fatal strikes
because the "'action' is mandatory" error never pointed at the READ tool → it repeated the same wrong call. Fixed:
workspace_track now detects a read-shaped call (no action, or a read verb like recent/changes/list/status) and
steers to `workspace(action='recent'/'changes')` instead of just listing write actions; a genuine bad-write typo
still gets the plain error. Test: test_bughunt_unit5_fs_exec.py::test_workspace_track_read_intent_steers_to_workspace.
DEPLOYED (listener 81458→82452, health ok). (Same turn also over-claimed the JJ Calendar API status and the
verifier correctly LATE-REFUTED it 90% — that's the verifier working, not a bug.)

### 2026-07-26 (later 7) — Improvements batch: 7 deferred LOWs + learning-health telemetry (#1) + Tor cache (#4) + wire-or-retire observability (#5)

Operator: "do all improvements except pg re-ingest and GAIA including the deferred lows."
DEFERRED LOWs (7, all fixed+tested): (1) helper_fetch_url_content now STREAMS the body with a byte cap and
checks headers FIRST (rejected PDF/oversized/non-200 never downloaded) — the "we count bytes ourselves" comment
was false; (2) 503/error retries rotate the CIRCUIT via socks_url_with_identity instead of
request_new_tor_identity, which on this box (control port 9051 closed) fell back to `brew services restart tor`
— bouncing the WHOLE daemon up to 2x/fetch and re-circuiting every sibling; (3) execute.py file-not-found heals
gated on _rerun_unsafe(command) — a compound with a mutating prefix (mkdir&&cat) no longer re-runs and adopts a
misleading "File exists"; (4) streamed trajectory records the EFFECTIVE visible content (fallback sentence), not
scrubbed tool-XML tag-soup; (5) read-budget occupancy now includes the ~24k injection reserve (_INJECTION_RESERVE
_TOKENS), not just history; (6) id-less native tool-call pairing keys by (name, occurrence) so two same-named
id-less calls don't collide/drop a result (extracted _reconstruct_tool_calls, now unit-tested); (7) intra-project
depends_on CYCLE guard in deps_satisfied (was silently reporting project_done on a plan with open cyclic work).
6 fetch tests updated to the streaming+circuit contract via a new conftest make_streaming_resp/make_httpx_stream_client.
IMPROVEMENT #1 (learning-health telemetry — the top eval rec): new core/learning_health.py (pure, defensive,
tested) → introspect action='learning' + scripts/learning_health.py. Aggregates lessons (outcome arms, hit-rate,
prune candidates), competence (domains crossing the 20-obs gate), episodes (field coverage), calibration
(Brier, entropy-distinctness), auto-skills, activity, and a COGNITIVE WIRING section. LIVE SIGNALS it surfaced:
FAILURE arm = 0 fail-only lessons (demotion loop likely inert on this 35B); calibration entropy = 2 distinct
values, w_entropy=0 (unlearnable — the streamed-path calibration fix from later-6 should now diversify it);
episodes 0/165 context coverage (predate the later-3 fix). This is the "measure before building more" instrument.
IMPROVEMENT #4 (Tor cache): the 300s success-only cache already existed; added _norm_cache_key (near-dup queries
share a key), cap 64→128.
IMPROVEMENT #5 (wire-or-retire): INVESTIGATION FINDING — the eval's "write-only, retire for RSS" premise is
mostly overstated. calibration/confidence feed the telemetry + idle refit; selfhood is read by introspect and its
stores are compaction-bounded (2MB/512KB) — NOT write-only, retiring would break readers. Only self_consistency
is genuinely inert (no caller). So rather than retire (a product decision, risky), made the wiring OBSERVABLE via
the telemetry's COGNITIVE WIRING section so the operator decides with data. self_consistency flagged as a
retirement candidate — left for the operator to delete.
NOT DONE (per operator): PG re-ingest, GAIA. #6 agent.py decomposition DELIBERATELY LEFT — the eval ranked it
last, it was stopped pending a real reaching-defs pass, and doing that 15.8k-line hot-path refactor autonomously
is the silent-cross-turn-bug risk the operator flagged. Tests: test_bughunt_fixes_2026_07_26b.py (deferred LOWs),
test_learning_health.py (6). Docs: introspect/search/execute/planning. DEPLOYED (listener 70296→75294, health ok);
full suite 9290 passed / 0 failed; functional_live_test 32/32 (streamed turn incl.); headless
scripts/learning_health.py verified against live stores. Operator next step: run `introspect action='learning'`
(or the script) periodically — it's the instrument for the ~2-week keep/kill watch on the learning stack.

### 2026-07-26 (later 6) — Six-agent bug hunt (least-audited systems) + improvement eval → HIGH/MED batch FIXED

Operator: "do a bug hunt on the systems that need it most, and evaluate for major improvements." Six parallel
read-only agents on the systems least-recently deeply audited (turn-loop core, web/search/browser, execution/
coding-correction, verifier/self-improvement, project-autonomy) + one improvement strategist. ~22 verified bugs
(3 HIGH, 9 MED, 10 LOW); every prior audit's fixes re-confirmed still holding. Agents told the §4B catalogue is
STALE (verify-before-report) — no re-report noise this time. Orchestrator verified every HIGH at source.
CROSS-CUTTING THEME: the streamed (default web-UI) path keeps diverging from _finalize_and_return — 3 separate
findings were one root cause. FIXED BATCH (operator "proceed"):
• Streamed-path cluster (agent.py): extracted _record_calibration_safe → called from finalize AND the streamed
  drain (streamed recorded ZERO calibration samples → fit trained on non-streaming subset; and only the stream
  path has real logprob entropy, so w_entropy was unlearnable — both closed); streamed post_mortem gained the
  not-forget_was_called gate (tombstone resurrection on the common path, incomplete fix from later-3); streamed
  drain now reads a req_id-tagged project-work SNAPSHOT captured under the semaphore (was reading/clearing
  process-global current_project_id + accumulators live, after release → misattribution/loss under concurrent
  turn B); late-refute filing uses the snapshot pid too.
• Calibration composite-leak (confidence.py): ConfidenceReading.pre_penalty_composite — record the prediction
  BEFORE the outcome penalty (penalty is a function of the label → Brier read optimistically on negatives).
• Constraint-gate sibling bypass (projects.py, HIGH): per-CALL audit cache reused the first task's verdict for
  every sibling in a bulk close → a violating sibling with different files closed DONE UNAUDITED. Now cached by
  frozenset(files).
• _op_ok substring (coding_executor.py): failed replace on a *success*-named file counted as applied → leaf DONE
  unchanged. Anchored to startswith; sibling _looks_like_write_error "security error" substring nit fixed too.
• Darkweb dead-engine (darkweb_search.py, HIGH): Ahmia (2/3 default engines) serves JS-only template (live-
  confirmed, 4727 bytes, 0 links) → silent empty-parse. Added 200-with-0-results diagnostic + JS-only detection
  + GHOST_ONION_ENGINES pointer; charset-aware decode (was force-utf-8 → mojibake on Cyrillic/CJK).
• Search (search.py): per-minute circuit bucket → 100ms (same-minute retries rode dead exits); race pool 4→8
  waves (isolated, safe); formatter .get(k) or default None-trap.
• Project-autonomy (advancer/cleanup): inter-project dep deadlock (deleted/archived/failed dep blocked forever;
  now cleared-with-warning, paused still blocks); partial keep-set now rescues binary deliverables (.db/.pt/.npy
  — were deleted at completion); autoadvance postcondition-demoted leaf no longer logged "completed".
Tests: test_bughunt_fixes_2026_07_26b.py (14) + test_projects_tool_review_fixes.py (+1 HIGH). Docs: 9 pages.
DEFERRED LOWs (logged, low value): id-less native tool-call collision, streamed-trajectory tag-soup, read-budget
injection omission, depends_on intra-project cycle, darkweb helper full-body-buffer, Tor control-port restart.
IMPROVEMENT EVAL (top 5, ranked leverage÷effort — NOT yet done): #1 learning-health telemetry surface (the FIX
for the calibration streamed-path bug was a prerequisite — you couldn't measure it before); #2 GAIA run (operator
huggingface login); #3 PG-manual re-ingest (96.7% of vector store, 100% corrupt); #4 Tor search reliability;
#5 wire-or-retire the write-only cognitive subsystems (selfhood/PRM/confidence consumers dead-gated). Live signal
worth noting: the outcome-gated FAILURE arm has produced ZERO lesson demotions — may be inert on this 35B.
DEPLOYED (listener 62438→70296, health ok); full suite 9278 passed / 0 failed; functional_live_test 32/32
(streamed turn included — confirms the streamed-path refactor serves correctly). WATCH: once #1 telemetry lands,
verify the streamed calibration samples now carry real (non-0.5) entropy and the fit's w_entropy actually moves.

### 2026-07-26 (later 5) — Functional live-test runner (scripts/functional_live_test.py)

Operator: "make functional tests against port 8000 ensure everything works correctly." NEW: a standalone runner
that hits the LIVE agent (no in-process mocks — tests/ already covers that) and validates the API layer + the
audit-fixed behaviors end-to-end. Non-destructive (nonce-tagged, self-cleaning; verified 0 durable pollution).
Two tiers: CORE (deterministic, no LLM — health/config/nodes, auth 403, malformed-body→4xx-never-500,
version/tags, memory-delete absent→clean-409, sessions CRUD, workspace-save + scratchpad-snapshot-is-a-dict via
in-zip session.json read = the export_state fix) and LIVE-LLM (real turns, tolerant/soft: PONG coherence,
manage_skills list names news_headlines+generate_password with no traceback, in-conversation recall). Result vs
prod (listener 62438): 32/32 pass, agent healthy after. Gotchas baked in: workspace zip is ~25MB (whole sandbox)
so DON'T re-upload it — read session.json out of the saved bytes instead; recall nonces must be ALPHABETIC (the
model normalizes a digit suffix, "zephyrine123"→"Zephyrine", reading as a false miss). Flags: --core (skip LLM),
--strict (soft→hard), --base. Not in the pytest suite (needs live node/Tor/data-dir); run manually or from cron.

### 2026-07-26 (later 4) — Post-audit improvements #1 (§4B residuals) + #2 (dormant subsystems wired)

Operator: "proceed with #1 and #2" from the improvement shortlist.
#1 (§4B backlog): AUDIT FOUND MOSTLY-ALREADY-FIXED. Confirmed at source that the contention theme (warm_up/
keepalive off_main_only, jobs-collect mark_collected read-marking), the router boot landmine (own try + retrain
fallback), the post_mortem transient-requeue re-raise, the streamed mid-stream fail-open (chunk_data.get("error")),
the idle code-gen max_tokens=4096, and the scratchpad injection cap (6000) were all fixed in the 2026-07-20/22
passes and just never struck from the §4B catalogue — the catalogue is STALE, not the code. ONE genuinely-open
item fixed: AcquiredSkillManager was constructed fresh at every call site, each with its own RLock → lock
serialized nothing, concurrent telemetry could lose a failure_count increment. Added AcquiredSkillManager.
get_shared(base_dir) — one process-wide instance per resolved path (upgrades a cached None memory_system);
all 4 call sites switched. Regression: 8 threads × 60 increments, zero loss.
#2 (dormant subsystems, the real value): (a) _record_episode_safe now populates context + cluster_id — all 165
live episodes had them EMPTY, so search_recoveries weighed only outcome text, consolidation grouped everything as
"general", and injected lines rendered "[]". Cheap (no extra LLM call): cluster_id = domain of first substantive
tool (metacog._domain_for_tool, aligns with competence taxonomy), context = tool-chain trace + explicit
"recovered after failures in: …" signal (what makes a turn detectable as a reusable recovery for the System-3
loop we wired in later-3). (b) competence.get_context_string() (39 live cells, computed-but-rendered-nowhere)
now injects into the prompt continuity blocks, gated on _COMPETENCE_MIN_OBS=20 total obs so a cold profile
doesn't inject small-n noise. Tests: TestSharedAcquiredManager (3), TestEpisodeFieldPopulation (2),
TestCompetenceContextString (2) in test_memory_audit_fixes_2026_07_26.py. Docs: acquired_skills/episodes/
competence pages. NOTE: §4B catalogue below is stale — treat its line items as "verify before fixing", not
"open". DEPLOYED (listener 60937→62438, health ok); full suite 9263 passed / 0 failed. Watch over the next
~2 weeks whether the competence block changes planner tool-choice and whether episode context/cluster improve
search_recoveries hit-rate (both now have real data to work with, unlike before).

### 2026-07-26 (later 3) — Memory-substrate audit: ALL ~60 actionable findings FIXED (7 clusters), tests+docs done

Operator authorized unattended fix-everything after the (later 2) scan. Every cluster fixed, unit-tested
(tests/test_memory_audit_fixes_2026_07_26.py, 59 tests + pinned-test updates elsewhere), and documented (dated
"Audit fixes (2026-07-26)" sections across 19 docs pages). By cluster:
• SPLITTER: helpers.py recursive_split_text re-attaches separators to EVERY fragment (whitespace seps were
  dropped → word fusion in all chunks) + chunk_overlap now honored on the separator path, carry CAPPED at
  chunk_size//4 (uncapped, the overlap-clamped-to-size-1 case — semantic_split_text long headers — advanced one
  fragment per chunk: quadratic blowup that spun the suite 36 CPU-min mid-session; cap guarantees ≥3/4 progress
  per chunk). Existing corpora stay corrupt until re-ingested; PG manual source PDF is GONE from disk, so
  re-ingest needs a fresh download (postgresql.org pdf) — deferred to operator, corrupt-but-answering corpus
  left in place deliberately.
• ATTRIBUTION: get_playbook_context resets last_playbook_triggers up front + stamp_triggers=False for dream sim
  (new last_sim_triggers channel) and ReadOnlySkillMemory (façade also blocks quarantine_lesson; _WRITE_HINTS
  gained "quarantine"); record_helpful_retrieval idempotent both orderings; PASSED flush moved before the corpus
  direction guard; _surfaced_lesson_triggers merges bus skill-tier survivors into the outcome arms (turn_id
  guard); belief-revision probe record_retrievals=False + $nin scope + delete restricted to OFFERED ids;
  projects lookup-miss probe silent.
• VECTOR: bus routes identity facts through smart_update (plain add left the conflict guard caller-less; live
  dupes); forget sweeps type-scoped ($nin document/episode/skill/acquired_skill + belt) — was deleting doc
  chunks via the distance-ignoring literal override; is_name_memory narrowed to real name statements (the
  "user's"/"user is" prose net granted -20 absolute rank + fake MASTER SUMMARY label); search_items off-topic
  gate mins over query-batch distances only (identity-probe distances defeated the 0.42 floor); syntheses +
  acquired-skill embeds get timestamps (timestampless synthesis = first eviction victim); ingest dedup re-check
  after filename auto-resolution; bus publish failures surface as PARTIAL in remember/update_profile/learn_skill;
  last_hydration clobber guarded by turn_id; REM fetch drops unused embeddings; CancelledError no longer eaten.
• GRAPH/EPISODES: add_triplets shape-guards non-dict triplets (one malformed triplet aborted fact+profile writes)
  + relation-key fallback + logged per-triplet errors; functional expiry skips _EXPIRY_GENERIC_SUBJECTS
  (project/task/app/… — generic-subject rows made expiry cross-project destructive); episode cap eviction
  UN-INVERTED (was deleting the dream's pending input, fossilizing spent rows); streamed episode write gained
  smart_memory/forget gates; reconcile_vector_index wired at boot (bg thread in main.py); System-3 pivot now
  injects search_recoveries via SYSTEM_3_EPISODIC_CONTEXT (was built on both ends, wired on neither);
  frontier duplicate re-rolls no longer inflate templates[*].runs; cluster_id render fallback.
• SKILL STORES: compile_from_pattern sanitizes dotted minted names + no-ops on existing (re-graduation demoted
  APPROVED macros + wiped stats; register() merges-not-demotes as belt); composed defs shadow against the FULL
  acquired registry (routed-subset gap = advertised≠dispatched); create_skill refuses builtin/macro names;
  degraded status recoverable (success clears; changed-content re-create starts clean; retirement catches
  legacy zombies by status; dead ≥5-rule dropped); TDD gate parses exit banner (substring was spoofable by
  echoed banners); telemetry: stale-file failures counted, infra outcomes (SYSTEM ERROR/124/137) charged to
  NEITHER side (_acquired_skill_result_class); routing []≠None (all-stale hits degrade to advertise-all);
  manage_skills list marks degraded/proposed; miner requires outcome=="passed" + _MACRO_UNSAFE_PARAM_TOOLS get
  empty templates (was baking stale task_update writes verbatim); GraduatedSkillStore.remove() + non-monotonic
  confidence + deprecate/retain_monitor verdicts handled at the phase-2.6 caller.
• SMALL STORES: journal in-flight/overflow UnicodeDecodeError → _preserve_corrupt sidecar (was silent unlink);
  _dedup_key strips retries (requeue twin double-consolidation); explain_belief_change token-overlap matching
  (whole-substring was inert for multi-word messages); profile _degraded fail-closed on OSError (intact file no
  longer sidecar'd away/overwritten); fsync-before-rename in profile/contradiction/adaptive/competence;
  category+key required for profile writes on BOTH the consolidation and bus legs (junk <cat>.info keys);
  adaptive record() takes effective_threshold (cleared_bar mislabeling); scratchpad LRU skips dunder sentinels;
  export_state/restore_state (locked, namespace-preserving; legacy flat shape → global scope); self-play report
  namespace=None; pending_count gate failure logged.
• PROJECTS/WORK-LOG: update_task reopen tuple gained DONE (revive-task deadlock); artifact_added carries path
  (file_history blind to registrations — live 60/60 pathless); find_deleted_similar needs ≥2 shared tokens or
  identical sets (Chess Game→Chess Tutorial mislink); project_events per-type retention (300) for
  task_updated/project_updated/work_log; work_log now records all-failure turns (<tool>_failed counts) and
  drops off-project interludes via _request_relevant_to_project (fail-open; live: news-curl row on Jiu Jitsu).
FULL SUITE: 9256 passed / 0 failed (3 pinned-test updates: macro-miner fixture outcome→"passed",
forget-hardening FakeCollection.query grew the `where` kwarg, workspace-save route degrades to legacy flat
scratchpad read for doubles). DEPLOYED (listener 52163→60937, health ok). DATA REPAIRS DONE + VERIFIED:
15 stale contradictory graph rows expired in the deploy down-window (chess-v4 status/pid, generic
project/task/service/system status rows, webos ×2 → 0 live conflicts); 7 stale identity rows deleted via
/api/memory/delete (5+1 chess "currently playing" paraphrases, old home_lab twin — richer "nova (runs Gemma)"
kept); episode boot reconcile ran live on first respawn: twins 131→141 (remaining gap = shared-trigger dedup
by design). REMAINING (operator input needed): PG manual re-ingest — source PDF no longer on disk; corpus left
serving (corrupt-but-answering beats absent); repair = download postgresql-17 PDF into the sandbox, then
knowledge_base(forget→ingest_document) with the fixed splitter.

### 2026-07-26 (later 2) — Six-agent memory-substrate defect scan: ~67 verified findings, 10 HIGH (NO fixes applied yet)

Operator asked for a defect scan across ALL memory layers. Six parallel read-only auditors (vector/recall, lesson
playbook, small stores, graph+episodes+frontier, projects+ingest, skill stores), each requiring file:line evidence
+ live-store probes (ro-sqlite only); orchestrator spot-verified every HIGH at source. Highlights by damage:
• INGEST (live-corrupt): utils/helpers.py:404 `recursive_split_text` drops whitespace separators on rejoin — ALL
  7,131 postgresql-manual chunks have words fused across line breaks ("DocumentationThe PostgreSQL…"); overlap=150
  honored only on the no-separator path. Fix + re-ingest = biggest RAG win available.
• LEARNING-LOOP POISONING: skills.py:1671 `last_playbook_triggers` never cleared on empty branches (early return
  precedes stamp) → outcome arms/hydrated_lessons/credit book stale or sub-agent/dream-sim trigger sets against
  wrong turns; readonly.py:122 ReadOnlySkillMemory lacks `quarantine_lesson` in _MUTATORS + no record_retrievals
  override (found independently by 2 auditors; drift-guard test _WRITE_HINTS lacks "quarantine" prefix). Plus:
  judge-after-inline double credit (skills.py:1166), PASSED flush starvation (agent.py:6302), bus-tier lessons
  invisible to FAILURE arm (agent.py:6341), belief-revision search phantom-credits + unscoped deletes (agent.py:4654).
• VECTOR ROT/LOSS: bus.py:1134 update_profile leg is plain add — smart_update (whole subject-conflict guard) dead
  in prod, live DB shows duplicate/contradictory identity rows; tools/memory.py:751 unified_forget sweep is
  type-unscoped w/ distance-ignoring literal override → deletes doc chunks/episode/skill twins (96.6% of rows are
  chunks); dream syntheses stored w/o timestamp → first evicted (dream.py:1654).
• GRAPH/EPISODES: graph.py:164 `t.get("subject")` outside per-triplet try — one malformed LLM triplet aborts the
  whole smart-memory task incl. fact+profile writes; episodes.py:208 cap eviction INVERTED (deletes unconsolidated
  pending input, fossilizes spent rows; latent at 165/500); streamed finalize writes episodes ungated by
  smart_memory/forget (agent.py:14740); System-3 recovery loop built on both ends, wired on neither
  (episodes.py:691 + prompts.py:1035); generic-subject `project HAS_STATUS` rows make functional expiry
  cross-project destructive (graph.py:47); 12 unique-trigger episodes lack vector twins (reconcile still uncalled).
• SKILL STORES: phase-2.6 re-mint demotes APPROVED macros to proposed + wipes stats (agent.py:3583 + unconditional
  register()); minted names bypass validation (the dotted auto.generic.* macro seen live today); shadowing
  one-directional → advertised def ≠ dispatched tool under routing filter (registry.py:709); "degraded" status is
  a one-way trap (no path back to active, re-create preserves it); skills_auto verifier is a tautological
  threshold re-check, confidence monotonic, deprecation unreachable.
• SMALL STORES: journal in-flight/overflow UnicodeDecodeError → treated as empty then unlinked/overwritten (silent
  loss, journal.py:329/373); explain_belief_change requires whole-message substring → inert (contradiction_log.py:190);
  profile.py lacks the _degraded fail-closed path its 4 siblings have; category-only profile updates mint junk
  `<cat>.info` keys (live: cli_tools.info); update_task reopen tuple missing DONE (projects.py:754); work_log
  attribution writes off-topic turns into project journals (live: "get me the news" on Jiu Jitsu Calendar);
  all-failure turns write NO work_log; artifact_added payload pathless → file_history blind to registrations.
Full ranked list with failure scenarios delivered in-session (2026-07-26); nothing fixed yet — awaiting operator
prioritization. Suggested order: splitter+re-ingest → attribution cluster → forget-scope+bus-dedup →
graph-abort+episode-gate → skill-store demotion/degraded-trap → rest batched per file.

### 2026-07-26 (later) — Duplicate skill embeddings: "injected 3" vs 2 registered skills

Operator: routing log said `injected 3: news_headlines, news_headlines,…` while `manage_skills` listed 2 skills.
Registry (one entry per name) is authoritative; the routing line counts vector HITS. Read-only sqlite scan of the
live chroma.sqlite3 confirmed: `news_headlines`×2 (07:23:07 and 07:26:08 — re-save with edited description) +
`generate_password`×1 = 3. Root cause: `save_skill` embeds on content change but never removed the name's previous
embedding — the content-hash dedup only stops re-embedding IDENTICAL content, so every edit added one embedding
forever. (Also confirmed the morning's orphan sweep worked live: `naftemporiki_headlines` gone from the store.)
Impact beyond logs: each duplicate wastes one of the routing query's n_results=15 slots; the advertising loop
itself was safe (registry iteration + membership check → never offered a tool twice). Three-part fix, DEPLOYED:
(1) `save_skill` delete-before-add with the now-working `$and` filter — an edited skill holds exactly one slot;
(2) `purge_orphaned_skill_embeddings` extended to collapse same-name duplicates (keep the embedding whose document
matches the registry's current description, else newest insert; orphan handling unchanged, snapshot-order guard
kept) — back-fills historical dupes on the next manage_skills call; (3) routing candidate list in `registry.py`
deduped order-preserving so "injected N" counts distinct skills. Tests: 5 new (save replace/no-churn, purge
collapse ×2, routing dedupe) + 4 older tests re-pinned (save_skill's own delete now precedes theirs — reset_mock
after save). Docs: `docs/tools/acquired_skills.html` (new "Duplicate embeddings per skill" section).

### 2026-07-26 — Acquired-skill vector deletes never worked (flat multi-key where) + orphan-embedding sweep

Operator hit `Failed to remove skill 'naftemporiki_headlines' from vector memory: Expected where to have exactly
one operator` on a manual delete. Root cause: both `delete_skill` and `retire_degraded_skills` in
`tools/acquired_skills.py` passed Chroma `where={"name": ..., "type": "acquired_skill"}` — Chroma rejects flat
multi-key filters (must be `{"$and": [...]}`, the pattern `memory/skills.py:519` already used with a comment
saying exactly this). So the vector-store leg of skill deletion has NEVER worked: manual deletes warned (the line
the operator saw), the retirement path swallowed it silently (`except: pass` — now logs a warning). Every skill
ever deleted/retired left its embedding behind. Orphans never advertised phantom skills (semantic routing in
`registry.py` filters candidates against the live registry) but crowded the routing query's `n_results=15` slots.
Two-part fix, DEPLOYED (listener 12024→50896, health ok): (1) both call sites → `$and` form; (2) new
`AcquiredSkillManager.purge_orphaned_skill_embeddings()` — reconciles type=acquired_skill embeddings against the
registry, deletes orphans by id, vector snapshot taken BEFORE registry read so a concurrent save (registry write
precedes embed) can't be misread as an orphan; wired into `tool_manage_skills` beside the retirement sweep
(back-fills the strays whose registry entries are long gone — no per-skill path could ever reach them). Watch for
`Purged N orphaned acquired-skill embedding(s)` on the next skill list/delete. Regression guard: tests capture the
`where` passed to the mock collection and run it through `chromadb.api.types.validate_where`, so a mock can never
again bless a shape real Chroma rejects (that mock/real gap is how this shipped). Tests:
`test_acquired_skills.py` (delete shape + 5 purge tests incl. manage_skills wiring), `test_acquired_skill_retirement.py`
(retirement now asserts exact `$and` shape, was only `assert_called`). Docs: `docs/tools/acquired_skills.html`.

### 2026-07-25 (later 5) — notes.info refill closed: workflow-cue backstop + sink-write guard + sink log level

Operator: "i still got 'profile capped — notes.info hit the 3-value cap'". Real recurrence, not stale code: the
serving process (boot 10:18:21) postdated the (later 2) fix (agent.py mtime 10:17:33), yet by 10:22 the freshly-
scrubbed ring held 3 NEW theme-toggle chatter values. Root cause: the post-boot **hippocampus drain of 14 buffered
journal items** (from the pre-restart dark-theme session) re-scored the episodes — the model inflated to 0.90 again
(rescored prompt ignored, as (later 2) predicted) — and the paraphrased facts ("…as previous attempts failed
verification / were refuted by a verifier / verify against specific project IDs and visual evidence") carry **no
12-hex id and no bound title** (drain runs unbound anyway), so `_is_tracked_project_state`'s two store checks could
never fire. Three-layer close in `run_smart_memory_task` + `profile.py`:
1. **`_WORK_STATE_CUE_RE`** in the backstop: verifier/refuted/failed-verification/previous-attempts/"project IDs"
   vocabulary = the agent describing its own loop, never a durable user fact; checked BEFORE the store lookups
   (store-independent), preference-exemption still first. Screens all 3 live-leaked values verbatim.
2. **Sink-write guard**: a `profile_update` naming NEITHER category nor key (the notes.info default shape) is
   dropped before the profile write — the fact already stores in vector memory on the same path, so the sink write
   was pure duplication churning the ring every consolidation. Well-formed updates flow; dropped ones degrade to
   `auto` type (same convention as the non-dict shape).
3. **Sink rotation → debug** in `ProfileMemory._bounded`: a fixed 3-ring rotating at its designed cap is by-design
   routine, not the context-leak signal — WARNING stays only for real merged keys hitting the 8-cap (chat-noise
   rule: operator stream = actionable only).
Live profile scrubbed post-deploy (3/3 slots were theme chatter again). Vector store purged too (operator-asked,
same session) via `/api/memory/delete`: the surviving theme-toggle near-dup + 3 OLDER identity-typed "user is
working on Jiu Jitsu Journal…" facts from the pre-fix session (found by read-only sqlite scan on chroma.sqlite3 —
NEVER a second PersistentClient, and probe the DELETE endpoint only with substrings you've already verified unique:
a broad probe deleted an unintended fragment — recovered because it turned out to be an ORPHANED lesson twin, no
JSON playbook entry, the very orphan that morning's "no JSON twin to bump" dedup-skip named, so net hygiene).
2 episode-typed utterance records about the project deliberately KEPT (legitimate history, not identity). Tests:
`test_project_state_memory_hygiene.py` (+3: live-leaked values verbatim, storeless cue screen, durable/preference
immunity), `test_agent_smart_updates.py` (+2: sink-bound dropped + explicit slot still writes),
`test_memory_store_durability.py` (+2: sink rotation silent, generic cap still WARNs). Full suite 9129 green.
Docs: `docs/memory/profile.html` (log levels + "Project-state screening + sink-write guard"). NOTE for the earlier
suite runs: this shell exports FORCE_COLOR=3 → 2 pretty_log non-TTY color tests fail on ENV, not code; run with
`env -u FORCE_COLOR`.

### 2026-07-25 (later 4) — PROJECTS full re-evaluation: findings LOGGED (fixes pending operator go-ahead)

Post-RELEASED-lifecycle design review (my targeted pass + independent full-surface review agent; both converged).
Live population: 9 projects (6 DONE, 2 RELEASED, 1 ARCHIVED); only 2 carry manifests.

**BUGS (ranked):**
- **P1 NEEDS_USER trap**: rollup can enter NEEDS_USER but nothing returns to ACTIVE (reopen tuples cover
  DONE/FAILED/PAUSED only; tool status enum omits ACTIVE; advance_once refuses non-ACTIVE) — answered questions
  strand the project forever. Same class as the 2026-07-11 DONE-reopen deadlock, missed for NEEDS_USER (+BLOCKED).
- **P2 RELEASED not locked at the store + archive→resume strips it**: `_maybe_rollup_project_status` locks only
  {DONE, ARCHIVED} — a store-level task write on a RELEASED project rolls it to DONE and FIRES THE SWEEP on a
  released workspace; archive(RELEASED)→resume lands ACTIVE (attestation + all three guards silently vanish,
  stale dossier/RELEASE.md still claim immutability). Needs: RELEASED in the rollup lock, resume restores
  prior status (or refuses), and an explicit `unrelease` policy.
- **P3 duplicate-create refusal loop**: `_TERMINAL_PROJECT_STATUSES` omits RELEASED → same-title `create` REUSES
  the released project (writing metadata via store, bypassing the choke) then steers the model to task_add —
  which the released guard refuses. Contradictory guard pair; should steer to create_version.
- **P4 no unregister path**: no delete_artifact / manifest-entry removal anywhere → a renamed deliverable
  permanently fails the release rehearsal with NO tool-side repair.
- **P5 double-fork collisions**: create_version has no fork-exists check and bypasses the duplicate-title guard
  → two "X v2" both on parent-port+1. No children index / family listing anywhere.
- **P6 services orphaned by lifecycle**: hard delete rmtree's under a running service; archive of a released
  project leaves its services running; failed release leaves rehearsal-restarted services up.
- Also: manifest dream-backfill starves legacy projects (top-2-recently-updated sliced BEFORE checking need —
  my 2026-07-24 code); digest blind to autoadvance_failed/budget_exhausted/project_reopened + all 4 release/
  version event types AND never scans RELEASED/PAUSED/BLOCKED/ARCHIVED projects; PAUSED not rollup-locked;
  `execute` bypasses the released write-guard (shell redirection); tool `_briefing.task_tree` uncapped (context
  hazard ≥50 tasks); `get` dumps the raw row; RELEASED switch/status still return dev-style briefing (view
  disagreement with the runbook mode); release/create_version don't title-resolve; rename breaks the `vN` title
  convention; sweep recovery registers PROJECT_MAP.md/RELEASE.md as deliverables.

**MISSING FEATURES (justified shortlist):** version-family listing + children index; release health-check /
re-rehearsal action (dossier rot detection); explicit unrelease→DONE (dossier retained); deliverable/manifest
unregister + rename reconciliation; service-lifecycle coupling (stop on delete/archive, optional start-on-resume
for released); task delete/reorder surface (store.delete_task exists, unexposed); cross-project work_log/
deliverable search; inter-project dependency edges; template/clone-without-lineage; notify-severity for
release_rehearsal_failed (actionable-event class per chat-noise policy).

Full detail in this entry + the review transcript.

**ROUND 1 FIXES SHIPPED (same day — operator: "proceed with round 1"):**
- **State machine (P1/P2 + PAUSED)**: rollup lock now {DONE, ARCHIVED, RELEASED, PAUSED} — a store-level task
  write can no longer roll a released project to DONE (and fire the sweep on an attested workspace) or complete a
  deliberately-paused one; the rollup gained a **back-to-ACTIVE branch** (open work reappearing on a
  NEEDS_USER/BLOCKED project rolls it ACTIVE — the answered-question trap is closed); reopen tuples on
  add_task/update_task now cover NEEDS_USER + BLOCKED (ARCHIVED/RELEASED still never auto-resurrect);
  **archive stashes `metadata.archived_from` and resume RESTORES it** — un-archiving a released project returns
  it to RELEASED with guards intact (legacy archives without the stash restore to ACTIVE as before). CONTRACT
  CHANGE: `test_add_task_never_reopens_archived_or_needs_user` updated (NEEDS_USER now reopens, rationale in-test).
- **Versioning (P3/P5)**: same-title `create` over a RELEASED project now steers to create_version (no more
  reuse-then-refuse contradiction, no store-bypass metadata writes); `create_version` is **idempotent** — an
  existing non-archived fork is returned with `existing_fork: true` (double-fork title+port collision closed);
  new `store.list_children`; `list` display annotates `[vN, fork of <parent>]`.
- **Unregister path (P4)**: `store.unregister_file` + tool action `unregister_file` (released-guarded) removes
  stale deliverable rows + manifest entry + re-renders PROJECT_MAP.md — renamed files no longer block release
  forever; the file also leaves the cleanup keep-set (correct for renamed-away paths).
- **Backfill starvation + digest (mine + M1)**: `_backfill_file_manifests` iterates ALL non-archived projects and
  counts only CONTRIBUTING ones against max_projects (the 6 legacy DONE projects are now reachable); digest
  `_RELEVANT` + renderer gained the milestone events (project_released/version_forked/release_rehearsal_failed/
  autoadvance_failed/budget_exhausted/project_reopened) with a "Milestones:" section, and the candidate scan now
  includes RELEASED projects (version_forked logs on the parent — was structurally unreachable).
Tests: `tests/test_projects_round1_fixes.py` (9) + contract update. Full suite green; deployed.
**ROUNDS 2+3 SHIPPED (same day — operator: "proceed with round 2, then round 3"):**
- *R2a services*: `_stop_project_services` on hard delete (before rmtree — no more orphaned processes under a
  vanishing workspace), on archive (retiring retires services), and on a FAILED release rehearsal (don't leave
  half-started services of a not-released project up). Rehearsal failure now also pretty-WARNINGs.
- *R2b hardening*: release chmods the workspace tree a-w (`set_workspace_readonly` / path-direct `_chmod_tree` —
  hard delete restores +w BEFORE rmtree since the DB row is already gone by then; create_version restores +w on
  the fork because copytree copies mode bits); `execute` gained `_released_shell_block` — a command referencing a
  RELEASED project's path AND carrying a mutation token (rm/mv/redirect/sed -i/…) is refused with the
  create_version steer, read-only shell untouched. Closes the shell bypass (review M8).
- *R2c views*: `_briefing` returns a RUNBOOK shape for RELEASED (release + versions list + routing note; no task
  tree, no "files you write live HERE"); switch/resume keep the runbook note; `task_tree` render capped at 3500
  chars (the ≥50-task 100KB tool-result hazard); `get` returns a slim summary (id/status/goal/version/parent +
  briefing) instead of the raw metadata dump.
- *R2d polish*: release/create_version title-resolve (+ joined _TITLE_RESOLVABLE so an explicit title beats the
  current-project auto-fill); `kind` default "" so `update kind=GENERAL` finally works (old default made intent
  indistinguishable).
- *R3 features*: **`unrelease`** (the sanctioned demote: →DONE, dossier RETAINED + stamped, workspace +w;
  re-release bumps the dossier **revision** — fork numbering keeps "version"); **`verify_release`** (re-runs the
  rehearsal + reports drift vs the dossier services; logs `release_health`; WARNING on degraded; never demotes);
  **`task_delete`** exposed (store.delete_task had zero callers; released-guarded).
Tests: `tests/test_projects_round23_fixes.py` (9; incl. chmod/fork-writability, shell-guard allow/block, runbook
switch view, unrelease→re-release revision bump, health-check drift, task_delete guard). Two found-in-test bugs
fixed: hard-delete's chmod restore ran after the DB row was gone (path-direct now), and the drift test needed
dir +w for unlink. Full suite green; deployed.
**LIVE-VERIFIED (v2 release health, 2026-07-25 09:53).** Two independent paths: (1) natural request ("is v2 still
runnable?") → the model verified MANUALLY (restarted jj-journal, browsed, HTTP 200) and reported operational; (2)
explicit `action=verify_release` → **healthy** ("jj-journal: up · port 8100 reachable", drift: []), `release_health`
event logged, status untouched (RELEASED). **Deploy-hygiene lesson caught en route**: probe 1 hit "unknown action" —
the serving process (81880, up since 09:32) predated rounds 2/3 because the 09:48 "deploy" killed a `pgrep|head -1`
match that was NOT the listener, and the respawn health-loop then found the already-running old agent and declared
success. Correct deploy handle: **the pid from `lsof -iTCP:8000 -sTCP:LISTEN` (or `launchctl print`), never
pgrep head -1** — and verify the LISTENER pid changed, not just that health returns ok.

**FINAL ROUND SHIPPED (same day — operator: "proceed with the deferred features"):**
- **`action=search`** (`store.search_projects`): "which project touched X / used Y" across ALL projects —
  keyword-scored over title/goal (3×), deliverables + manifest descs (2×), ledger/research/work_logs (1×);
  ranked hits with kind-tagged snippets; empty-hit response steers to list/recall. Deterministic, no LLM.
- **`action=set_dependency`**: `metadata.depends_on_projects` (ids/titles resolved; self + transitive cycles
  rejected, depth-10 cap); **`advance_once` gates on it** — a dependent project reports "waiting on dependency
  'X' (STATUS)" until every dep is DONE/RELEASED, so the round-robin can sequence project chains; briefing shows
  DEPENDS ON with live statuses; `dependencies_set` event.
- **`action=clone`** (template/clone-without-lineage): fresh-titled project from any-status source (RELEASED
  sources copy fine — mode bits restored on the copy), carrying files/ledger/config(port+1)/manifest/constraints;
  provenance = `cloned_from` only (NOT a version fork — list_children ignores clones); deliverables re-registered
  on the seed task; auto-switch + PROJECT_MAP render.
Tests: `tests/test_projects_cross_features.py` (7). Full suite green; deployed (listener-pid-verified per the new
deploy rule); live search sanity on the real 9-project population.

**Still-deferred residue (small, non-blocking)**: cross-version journal chaining; PROJECT_MAP/RELEASE.md keep-set
formalization (safe by suffix today); sweep-recovery deliverable-classification of system .md files.

### 2026-07-25 (later 3) — Vision-claims refute class closed: claim-conditioned evidence selection

Diagnosis overturned the design: the 🌙→☀️ "vision claim" was never vision-only — the click result CARRIED the
post-click state as text ("button now shows ☀️") and the screenshot description confirmed it. The verifier refuted
a true claim because a failed-click retry + re-navigate + task close pushed the evidence-bearing outputs **outside
the newest-3 positional window** — the souvlaki class (2026-07-16) one level up. So the fix is not vision plumbing:
`_collect_verifier_evidence(claim_text=)` now scans a bounded deeper window (10 substantive outputs) and pulls the
older output that best overlaps the CLAIM's significant tokens (≥2, stopword-stripped) as a 4th evidence slot
(new 4-way budget split [0.4/0.25/0.2/0.15]; newest-3 positional behaviour unchanged when no claim/no hit;
deterministic, zero extra LLM calls). Call site threads `final_ai_content` as the claim. Genuinely-vision-only
claims (state visible ONLY in pixels, never in any tool text) remain a theoretical residue — none observed yet;
verify_visual exists if one ever does. Tests: 3 more in `test_project_state_memory_hygiene.py` (displaced-evidence
reproduction, legacy-window preservation, overlap threshold). Full suite green; deployed.

### 2026-07-25 (later 2) — Operator log-review trio: project-state memory hygiene + id-linkage evidence + release port seed

Operator flagged three log entries from the dark-theme turn; all three were real. (1) **User-memory pollution**: the
smart-memory prompt scored "PROJECT CONTEXT" at 0.8 and the model inflated to 0.9 → three project-status "facts"
stored as timeless user truths AND churned through the profile's 3-slot notes.info (100% of the user's notes became
project chatter, evicting whatever was there). Fix: prompt rescored (tracked-project state → 0.2 DISCARD; "the
project store owns these"; preferences learned during project work stay 0.9) + deterministic backstop
`_is_tracked_project_state` (12-hex token verified against the store, or bound-project title; preference-cue
sentences exempt) applied to BOTH the fact and the EFFECTIVE profile value (profile_up.value falls back to fact —
the fallback is screened too). Graph triplets deliberately untouched (project concepts belong in the graph). The 3
polluted notes.info values scrubbed on deploy. (2) **Category-error refute**: verifier refuted the correct close
because "task ID d8a307dd196f does not match project ID 6be718ab7dc3" — different entity types; the linkage evidence
(task_update confirmation naming the task id) was under the packer's 200-char bar. Fix: short bookkeeping results
carrying a 12-hex id are packed (linkage-evidence class). Noted consequence chain: that LATE REFUTED (90%) backfilled
the trajectory FAILED → the outcome-gated loop ticked `failed_retrievals` on lessons from a turn that actually
SUCCEEDED — false refutes now poison the failure arm, raising the stakes on verifier precision (`_OUTCOME_MIN_OBS=4`
bounds it). The second refute half ("evidence only shows initial state" for the 🌙→☀️ claim) is the vision-claims
class — screenshot-observed facts aren't in text evidence; logged, not fixed. (3) **Release port seed**: the dossier
rehearsal's discovered service port is now written to `config.port` when absent — so create_version's bump always
has a source (the first live fork served on v1's port for lack of one). Tests:
`tests/test_project_state_memory_hygiene.py` (9). Full suite green; deployed; profile scrubbed (3/3 slots were
chatter — nothing durable had survived their churn).

**v2 RELEASED with the FULL dossier (after one more found-bug).** First v2 release rehearsal STILL fell back to the
deliverables check: the live service registry stores workdir="/workspace" with the project path only in the COMMAND
(`cd /workspace/projects/<id> && node server.js`) — the workdir-only matcher missed it. Fixed
(`_project_service_entries` matches workdir OR command; test updated to the live registry shape) + "v2 v2" title
cosmetic. Re-released (store-level DONE revert, operator-authorized re-validation): **rehearsal took the service
path — "jj-journal: up · port 8100 reachable"** — dossier carries the verified start command + URL, and
`config.port=8100` seeded → a future v3 fork bumps to 8101. The release pipeline is now validated on both rehearsal
paths (deliverables + live service).

### 2026-07-25 (later) — RELEASED lifecycle: human-attested terminal state + rehearsed runbook + versioning

**Operator idea, evaluated + hardened, then built.** DONE = the agent's self-assessment; RELEASED = the human's.
Three pillars, each with a hard mechanism (not convention):
1. **`action=release`** (DONE + human command + model-written `directions` required) → workspace tidy → **deterministic
   release rehearsal** (ServiceSupervisor cold-restart + TCP port probe per service; deliverables-exist check for
   service-less projects — rehearsal-after-tidy also catches keep-set gaps) → on pass, freeze the **release dossier**
   (`metadata.release`: directions + VERIFIED service commands/ports + URLs + deliverables w/ manifest descs + config +
   rehearsal transcript) + render `RELEASE.md` (dual pattern like PROJECT_MAP.md) → status RELEASED. Fail → stays DONE
   with detail. Rollup can never set it; `update` rejects `status=RELEASED` ("earned, never assigned").
2. **Immutability, three layers**: tool-level choke (`_released_guard` on all mutating actions incl. autoadvance/
   research/cleanup, read-forms exempt, steer → create_version); store-level (add_task's reopen tuple excludes
   RELEASED); **file-write-path level** (`_released_write_block` in tool_file_system write/replace/delete/move/append —
   matches `projects/<12hex>/` in the target, one status lookup; registry lambda threads `project_store`). Rationale:
   the same-day audit measured the agent regressing working artifacts — the write path must refuse, not trust.
3. **`action=create_version`** (RELEASED parents only; DONE edits in place): forks "Title v(n+1)" — carries files
   (minus RELEASE.md/.services), ledger, config with **port bumped** (v(n) keeps running while v(n+1) develops; ledger
   seed line records the split), manifest, constraints, research; FRESH task tree (seeded w/ the change request) +
   work log; copied deliverables re-registered on the seed task (cleanup keep-set safety); lineage events both sides;
   auto-switch. Dossier/task-history/retrospective stay with the parent.
Plus **runbook briefing mode**: RELEASED briefing = RELEASE DIRECTIONS + verified SERVICE/URL lines + routing
("run → follow directions; change → create_version"), all dev scaffolding suppressed. `_briefing` gains `release` key.
New `ServiceSupervisor.list_entries()` accessor. Tool schema teaches both actions ("release ONLY when the USER
explicitly confirms the project works").

**Verification.** `tests/test_project_release_lifecycle.py` (13: gates, happy path incl. RELEASE.md, rehearsal fail,
service-rehearsal w/ stub supervisor, update/guard/add_task/file-write immutability, version inheritance incl. port
bump + keep-set re-registration + parent untouched, runbook briefing). Full suite **9086/0** (one fixture needed
`asyncio.run` — `get_event_loop` broke under the suite's loop state). Docs: `docs/memory/projects.html` "RELEASED"
section. Memory `[[project-accessibility]]` updated.

**First real v2 development turn (dark-theme task, same day).** The agent implemented the toggle in v2's
index.html (3 surgical replaces), started the service pointed at **v2's workdir**, and browser-verified the
toggle live (navigate → click → 🌙→☀️ → screenshot) before closing the task DONE with a real result summary; v2
rolled up DONE. **Immutability proven by checksum: v1's files byte-identical after the full turn; v1 still
RELEASED.** The same-day audit fixes fired visibly in production: the WEB-EXEC file:// gate declared itself
inconclusive (page calls fetch) and the verifier caught a task-ID inconsistency in the closing claim.
**Finding:** v2 served on :8100 — v1's advertised port — because v1 never recorded `config.port`, so the fork's
bump had nothing to bump (and the model ran service stop-all first; harmless here since v1's service was already
dead, but a real collision hazard when a released service is RUNNING). Follow-up: seed `config.port` at release
time from the rehearsal's discovered services so every fork has a bump source.

**LIVE-VALIDATED (prod :8000, the real Jiu Jitsu Journal).** (1) "I have tested it, release it" → model called
`action=release` with genuinely good user-manual directions → rehearsal passed (deliverables path: "all 6 present")
→ **RELEASED v1** + RELEASE.md on disk, dossier in metadata. (2) "add a dark theme toggle" → model, unprompted beyond
the briefing, reasoned "project is RELEASED, I need to create a version fork first" → `create_version` → **v2
(6be718ab7dc3)**: ACTIVE, lineage recorded, ledger+manifest carried, NO release key, seed task "Add dark theme toggle"
PENDING, 9 files copied WITHOUT RELEASE.md; **v1 untouched and still RELEASED**. Two noted minors: the service
registry was empty at rehearsal time so the dossier has `services: []` (the URL survives in the directions; the
verified-start-command slot fills on projects whose services are registered at release time), and the port bump
no-opped because v1's config never recorded a port (bump requires a config `port` key — by design).

### 2026-07-25 — Log-mined deficiency audit (statistical + behavioral) over the 33h window 07-23 21:02 → 07-25 06:09

**Method.** First real payoff of the durable-log overhaul: one statistical pass (error/warning dedup, timing anatomy,
loop yields) + one behavioral deep-read of full turn narratives (30 requests; deep-read 9). Findings ranked.

**A. VERIFICATION THEATER (SEV-HIGH — the headline).** The persistence-migration turn (`4e42973e`, 1058s, 40/40
turns) shipped BROKEN code at confidence 0.99: turn ended on budget exhaustion mid-verification (last click never
evaluated) yet outcome=ok; the async verifier **LATE CONFIRMED (90%) off a `file://` page load** — which cannot
exercise a fetch-backed API app; the decisive POST→reload→GET test never ran (the intended `curl … | python3` was
validator-BLOCKED and the agent substituted a weaker GET-on-empty-store check instead of retrying in an allowed
form — which `6e0922c8` proved works via base64 urllib). User then filed 2 bug reports the same evening
(reload-empties-page = the fire-and-forget cache init; broken nav = see B). Related: bookkeeping-only turns are a
**verifier blind spot by design** — `5ae632e6` hallucinated describe_file success (3 empty returns → "calls were
made successfully", conf 0.95, verifier skipped); `54a10d05`/`82e2c451` fabricated briefing specifics (OS version,
dates) caught only by LATE refutes. **Directions:** cap confidence on budget-exhausted/PARTIAL turns; WEB-EXEC
evidence for served API apps must not be `file://`; validator-block → retry same intent in allowed form; echo-back
verification for bookkeeping writes; specifics (dates/versions) must trace to current-turn tool output.

**B. SELF-INFLICTED REGRESSION, not caught (`6e0922c8`→`ffcebd96`).** Bootstrap refactor replaced `Router.init()`
with `bootstrap()` but left nav-listener registration in the now-orphaned `init()` — broke all top navigation; the
post-edit browser check loaded the page but clicked nothing. Next turn correctly diagnosed "Router.init() is
defined but never called!" — its OWN 24-min-old change. All 3 evening bug reports were fallout of the agent's own
preceding edits. **Directions:** after replacing an entrypoint symbol, grep for references to the old name;
post-edit verification must exercise ≥1 interaction.

**C. VERIFIER FALSE REFUTES burn repair rounds.** `82e2c451`: refuted "Two lessons learned" though list_lessons
showed exactly 2 (repair destroyed a mostly-correct reply, 68-char final). `d32e2429`: refuted "All 9 tasks
complete" though task_list showed 9/9 DONE (625-token think + 3 redundant calls chasing an unsatisfiable refute →
loop breaker). Signature matches `[[verifier-evidence-packer]]` truncation — audit the evidence packer BEFORE
blaming the judge. Also `40ed42be` +509s refute was defensible-but-redundant (~60s re-proving rg-proven facts).

**D. EFFICIENCY.** (1) Latency is **94–97% reasoning-token generation** on the big turns (536/569s, 1002/1058s,
474/488s); tools are sub-second. Every LLM call re-opens with a task restatement (4–6×/turn); design deliberation
reverses in-token (turn 17: 1832 tokens cycling 5 designs, then reversed anyway). (2) Malformed SEARCH/REPLACE
tax: ~15–20% of the big turn's wall clock (4 failed edit attempts incl. the `content==replace_with` corruption
guard + 2 marker-parse failures) — `[[native-tools-corruption]]` still a live tax. (3) `6e0922c8` thinking-loop:
n-gram abort at 9568 chars after ~5 repetitions of the same 3-hypothesis cycle; proposed the decisive experiment
in-think and NEVER executed it (re-read code instead; loop breaker ended the turn). **Direction:** a
proposed-in-think experiment should become the next tool call; >2 design reversals → forced written plan step.

**E. IDLE LOOPS (yield ≈ 0 tonight).** Dream: 8 cycles → **0 meta-memories every time**, "Auto-memory pool thin
(2)" every cycle (dream-seed starvation is total; 2–4 dedup-churned heuristics/cycle). Reflection: 7 cycles
re-walked the same 85 failures for **1** new reflection (no incremental watermark). Self-play recycled the SAME 4
challenges all night, all first-try/Δ=0.000/write=False — root cause: the frontier `record_run` KeyError meant
runs never recorded, so saturation never accrued; all 6 occurrences PRE-DATE the 07-24 fix → **expect rotation to
self-heal; verify next overnight.** Lesson-outcome stash→drain rate ~20% (5 stashes → 1 drain): most clean turns
never earn a decisive verdict → arms fill failure-biased; watch at the 2-week audit.

**F. INFRA OUTSTANDING.** (1) **Embedder/store mismatch STILL live** (24 FATALs at 21:31 — re-embed never run).
(2) **68 doomed-node breaker trips, 0 recoveries**: `192.168.0.20:8088` (52×) AND `http://nova:8088` (16×, post-
config-fix) — worker is configured by tailnet IP, so something registers the LAN identity at runtime; prime
suspect: swarm-advertise/alias storing the node's self-reported hostname (macOS daemon drops LAN SYNs —
`[[macos-daemon-local-network]]`, launcher comment has the tcpdump proof). Fix: canonicalize advertised URLs to
configured tailnet addresses. (3) **transformers overflow capture BROKEN**: 0 lines in durable log, new
`194308 > 131072` in .err — the logger-capture is bypassed (custom transformers logging or subprocess). (4)
`System 3 pivot failed:` logs an EMPTY reason. (5) resource_tracker semaphore-leak warnings ×22 (minor).

**G. What worked (calibration).** Guard stack prevented every failure from becoming corruption or a hang (but 3
user-facing turns ended BECAUSE a guardrail force-terminated them — compensation, not health). `ffcebd96` fixed
the nav bug correctly in 71s with click-verification. Frontier fix holds (0 post-fix failures). 0 hard-failed
turns. The new instrumentation (turn outcomes, idle summaries, hydration-judge, lesson-outcome) made this audit
take minutes.

**One-line summary:** mechanics are solid; the core deficiency is **verification theater** — high-confidence
success claims on work never end-to-end tested, a verifier that confirmed a broken deliverable off `file://`
evidence while refuting two true claims, and ~95% of wall-clock spent on often-circular reasoning tokens.

**FIXES (same day — operator: "proceed with all fixes"; 9 items, all shipped):**
1. *Budget-exhaustion confidence cap*: `FinalizeState.turn_budget_exhausted` (default False; set by the for-else
   PARTIAL path) → outcome_penalty 0.8 in the finalize confidence compute + calibration outcome 0.0 + Turn
   Outcome shows `partial (budget exhausted)` at WARNING. A 40/40-turn reply can no longer ship at C=0.99.
2. *file:// evidence gate*: `_execute_web_artifact` declares itself INCONCLUSIVE when a probed page calls the
   network (fetch/XHR/axios//api/) — a file:// load can't exercise it (fetch rejections don't even trip the
   uncaught-exception marker); the existing `_WEB_EXEC_SKIP_CONF_CAP=0.6` then caps any text-only CONFIRMED.
3. *Validator precision* (`tools/validators.py`): the curl-pipe deny rule split — shells stay fully blocked;
   interpreters block only when fed the pipe AS THE PROGRAM (bare `| python3` or `| python3 -`);
   `| python3 -m json.tool` / `-c` / script-file forms are DATA and now pass. Root cause of the substituted
   weaker check in the 4e42973e migration turn.
4. *Validator-block retry steer* (`tools/execute.py`): the SYSTEM BLOCK message now instructs retrying the SAME
   verification intent in allowed form (curl -o file + json.tool, or file_system+execute urllib) — never
   downgrade to a weaker check.
5. *Bookkeeping blind spot*: `_find_substantive_tool_for_verifier` now returns a bookkeeping tool whose content
   starts with Error/SYSTEM BLOCK/REJECTED — errors are evidence, so a success claim over three failed
   describe_file calls gets verified (and refuted) instead of skipped "by design".
6. *Evidence packer false-refute fix*: `_collect_verifier_evidence` now packs INFORMATIONAL bookkeeping output
   (≥200 chars or error-prefixed) — task_list/list_lessons data behind a claim reaches the judge; short
   state-change confirmations stay excluded (the 2026-04-19 `{"exited":…}` blast radius). Run-gate vs
   evidence-set deliberately split.
7. *Node-list boot validation* (`main.py`): six copy-pasted parse loops → one `_parse_node_list` helper that
   WARNS at boot on LAN IPs (192.168./10./172.16-31, non-tailnet) and dotless hostnames — the 68 doomed breaker
   trips were transient launcher states (192.168.0.20 = the old documented config; nova = a ~16:05–18:00
   intermediate edit); current config verified clean (tailnet). Misconfigs now announce themselves at boot, not
   after 68 failures.
8. *Self-identifying embedder FATAL* (`memory/vector.py`): the mismatch error now names chroma_dir + pid + cwd —
   the 24-FATAL respawn loop at 21:31 was undiagnosable post-hoc because the message never said WHICH store
   (main store verified healthy, 7376 docs + fingerprint; the 161-doc offender is not on disk under any known
   path — next occurrence will name itself). Blind re-embed deliberately NOT run.
9. *Oversized-payload signal owned* (`utils/token_counter.py`): transformers' sequence-length warning silenced
   at the counting layer (verbose=False — it escaped only to raw stderr and is noise where no forward pass
   happens); estimate_tokens now logs its own durable INFO when a payload exceeds the 131k window (the real
   signal: something built a 194k-token string). Plus `System 3 pivot failed: %r` (empty-message exceptions no
   longer log blank) and the **orphaned-symbol guard** (`tools/file_system.py`): a replace that REMOVES a
   function/def whose references survive appends a named warning to the tool result — the Router.init class.
Tests: `tests/test_audit_fixes_20260725.py` (14); full suite green; deployed. NOT fixed by design: the
efficiency cluster (reasoning-token latency, restatement preambles — model-behavior work, needs its own pass)
and idle-loop yield (self-play rotation expected to self-heal post frontier-fix; dream-seed starvation is the
known #4 blocker).

### 2026-07-24 (later 2) — Project accessibility: file manifest + per-file history + journal densification + selective loading

**Mandate.** Operator: on "resume 6a471d630e81…" the agent just re-reads code — nothing directs it where to
look; on a hundreds-of-files project that collapses. Goal: make a complicated project accessible to a smaller
model by breaking it into small pieces. **Diagnosis from the (new) durable log:** the model read `server.js`
just to learn what "the service" was, then spent ~80s of thinking re-deriving the module architecture a prior
session had already derived. Existing substrate (Explore-mapped): work_log journal GOOD but sparse (2 rows vs
8 autoadvance steps — autoadvance bypassed it), design_ledger real but free-prose, deliverables = bare paths,
briefing = push-everything-top-N, workspace TrackedFile substrate unwired ("0 tracked files").

**Built (all four phases, finish-the-half-built):**
1. **File manifest** (`memory/projects.py`): `metadata.file_manifest` {rel_path: {desc, role, ts}} — ledger/config
   idiom (bounded 60 files/200 chars, atomic cross-process, oldest-updated eviction); `describe_file`/`get_file_manifest`;
   path normalization shared with the artifact keep-set (`_normalize_rel_path` extracted verbatim);
   `register_file_artifact(description=)` fed by the coding executor (build summary) + task_update DONE (result
   head); `manage_projects action=describe_file` (tool schema teaches "record it the moment you learn it");
   dream-cycle `_backfill_file_manifests` (ONE batched off-main worker call, ≤4 files/cycle — caught by the
   `off_main_only` contention guard on first try, fixed); every write re-renders **PROJECT_MAP.md** into the
   workspace (atomic) so the map is greppable in-sandbox too.
2. **Per-file history** (`file_history(pid, path)` + `action=file_history`): newest-first journal slice for one
   file; BOTH sides normalized (live payloads mix bare names and absolute /workspace paths). "What happened to
   index.html?" now answerable without re-reading it.
3. **Journal densification** (`project_advancer._work_log_step`): autoadvance coding DONE/FAILED + generic-tool
   DONE now mirror into work_log (request prefixed `[autoadvance]`), closing the 2-vs-8 gap.
4. **Selective loading** (`prompts.build_project_briefing(request_text=)` + agent.py call site): RELEVANT TO THIS
   REQUEST section — deterministic stopword-stripped keyword overlap over manifest (top-3 files, each with its
   last history line) + deeper work_log window (beyond the newest-5). Zero LLM cost. Guard test pins that the
   exact live phrasing ("resume X and give me a status update") matches NOTHING (stopword list built from it).
   DELIVERABLES upgraded to `path — description` (+ undescribed nudge); tool `_briefing` gains additive `file_map`.

**Verification.** 4 new test files (28 tests: storage bounds/eviction/traversal, tool actions, briefing surfaces,
autoadvance mirror incl. file_history + manifest seed, relevance slice incl. false-positive guard); full suite
green. Docs: `docs/memory/projects.html` new "Project accessibility" section. Memory `[[project-accessibility]]`.

**LIVE-VALIDATED (same day, prod :8000) — including one bug found BY the durable log and fixed live.** First
live `describe_file` probe: the model called it 3× and "got empty" while believing it succeeded — root cause:
the shared `description` param's schema text said "Task description (task_add/task_update)", so the model didn't
pass it, and the handler's no-description path silently fell back to manifest read-back. Fixed: schema text now
covers describe_file; file_path-without-description returns an instructive steer (test pinned); the Project Tool
log line now carries `file=`/`desc=` for the per-file actions (it was undiagnosable as `action=describe_file`
alone). Second probe: manifest landed (2 entries with roles) + `PROJECT_MAP.md` rendered with undescribed-file
nudges. **Then the operator's EXACT original request ("resume 6a471d630e81 , start its service and give me a
task update") re-run: ZERO file reads, zero sandbox listings** (before: read `server.js` + `sandbox tree`) — the
model went briefing → start service in 28s, and when the verifier refuted its task-count claim it re-checked via
project-tool actions, not file re-reads. `turn outcome — ok · confidence 0.99 · tools: manage_services×3,
manage_projects`.

### 2026-07-24 (later) — Logging overhaul: the durable log now reconstructs a turn (3-agent audit → keystone + noise + content)

**Mandate.** Operator: logs must be (1) extremely readable for human monitoring and (2) extremely informative — "I often give Claude the log to figure out what the agent did." 3-agent read-only audit (turn/tools, cognitive/memory, infra/errors) + a cross-cutting real-log profile.

**Root cause (all 3 agents converged).** The two sinks were disjoint: `pretty_log` → stdout ONLY (pretty stream `~/Data/AI/Logs/ghost-agent.log`, truncated, wiped each boot; the DEBUG_MODE file mirror was off in prod), while `logger.*` → durable `$GHOST_HOME/system/ghost-agent.log`. Measured: the durable log had **0** turn narratives / verifier verdicts / tool results, and was **56% one repeated line** (`Semantic Toolkit Router injected 2 acquired skills` ×1096). Neither log alone could reconstruct a turn.

**Keystone fix — `utils/logging.py` `_mirror()`.** pretty_log now emits its FULL untruncated content (+ req-id + delta + level) to a dedicated file-only logger (`GhostStream`, propagate=False, carries only the file handler → never double-prints on stdout), on EVERY call incl. BEGIN/END/SECTION frames, regardless of DEBUG_MODE. The durable log is now a COMPLETE, plain-text, restart-surviving record; the pretty stream stays the truncated curated view. (Operator chose: mirror-to-durable + drop `--verbose` for a clean stream + fix code-side bugs / flag config bugs.)

**Noise (Tier 1).** Removed the redundant `logger.info` router twin (56% of the log) and made the surviving pretty line NAME the skills; watchdog start/cancel + composed-skill re-register → DEBUG; `PRM serve-inert` + journal-recovery WARNING → INFO (off the operator's WARNING+ stream).

**Content, not counts (Tier 2/3).** Verifier verdicts now carry their reasoning (CONFIRMED says WHAT); `SKILL ACQUIRED` names the fix; belief revision (was debug-only → dropped in prod) names the dropped facts; file writes report size; circuit-breaker RECOVERY + HALF_OPEN probe now log (node healing was invisible — only trips logged); self-play crash now NAMES self-play then RE-RAISES (contract: it deliberately propagates to the tick handler with the anchor moved by finally — pinned by `test_anchor_updates_even_when_self_play_raises`, which the audit's naive "add except" would have broken); `transformers`/`py.warnings` captured into both sinks (surfaces the 159k>131k context-overflow that escaped only to raw `.err`).

**Real bugs surfaced (operator: fix code-side, flag config).** Code: context-overflow now surfaced (transformers capture). CONFIG (flagged to operator, not changed): node `192.168.0.20:8088` circuit-flapping = a LAN IP an off-host node can't reach (should be tailnet, see `[[macos-daemon-local-network]]`); `FATAL embedder/store mismatch` (a store's 161 fragments in old MiniLM space vs bge-small → garbage retrieval, re-embed). Still open: `Frontier record_run failed: 'runs'` KeyError.

**Verification.** New `tests/test_logging_durable_mirror.py` (4) + updated `test_registry_skills`; full suite **9035 passed / 13 skip / 0 fail**. LIVE (redeployed via KeepAlive): the durable log now reconstructs a full tool turn end-to-end — request-started → hydration → named routing → **thinking** → **shell command** → exit code → **verifier CONFIRMED + reasoning** → confidence → finished. Docs: `docs/logging.html` (new "Two sinks + durable mirror" section). Memory `[[logging-convention]]`.

**Tail sweep (later same day) — 12 more items landed.** Turn-narrative: execute STDOUT on success (was "exit 0" only); consolidated **Turn Outcome** line (state+confidence+tools, non-streamed path); SQL text on `postgres_admin`; **Service** start/stop/restart pretty_log (mutating sandbox-service ops were silent); download URL no longer pre-truncated to 35 (starved the mirror). Cognitive content-not-counts: **hydration-judge** now names which surfaced memories the reply used; **quarantine** now logs the pulled belief+reason (was silent); **Skill Retracted** names the scrubbed triggers; **Graph Updated** shows the actual `s→r→o` triples. Infra: **complexity-router** decision logged (label+confidence+escalated+reason — the "why deep-reasoning fired" record); circuit-breaker recovery already done in the main pass. **Real bug fixed:** `Frontier record_run failed: 'runs'` was NOT the cluster (that's back-filled by `_ensure_cluster`) — it was a LEGACY per-template dict predating the "runs" key, so `tstats["runs"] += 1` raised KeyError; fixed with `.get`, regression test `tests/test_frontier_legacy_template.py`. Full suite green throughout.

**Idle-scheduler instrumentation (later — the last deferred item, now DONE).** The noise design: the per-cycle summary is **INFO → durable log only** (off the now-`--verbose`-free pretty stream) and fires **once per idle cycle that actually did work** (frequent all-cooldown ticks append nothing → stay silent), so no per-tick spam. In `_biological_tick`: a `_idle_ran` accumulator that each meaningful phase appends to (dream / self-play / postmortem / skills-auto) → one `logger.info("idle cycle: ran … (idle Xm)")` at tick end. **The flagged blind spot fixed:** self-play's dice roll was ANDed with the cooldown (`cooldown and _bio_roll(0.2)`) so "never fired" was indistinguishable from crash / lost-roll / not-idle — split into three branches, each logged (cooldown-skip DEBUG, dice-miss INFO, ran). Reflection + PRM/router already log their own result/SKIP so they're left out of the summary (would double-count). Tests: `tests/test_idle_scheduler_logging.py` (dice-miss logged + doesn't run; run emits summary); self-play C3 anchor contract still green.

**Final closure (later — operator: "close logging today"; ALL previously-deferred items now DONE):**
- *Cognitive content-naming*: **Dream Consolidated** now logs each ACCEPTED synthesis (`N fragment(s) → text`; before, only rejections logged — the stream showed what the dream refused, never what it built); the counts-only "Dream Complete" stays as an index line since every category now has per-event content lines beneath it. **RRF refit** logs observation count + changed cells with old→new deltas (was a silent hot-swap of the fusion matrix). **Hippocampus** names each consolidated item (`type: subject-head`). **Recall-threshold retunes** log old→new (inert under `--smart-memory 0.9`, but recorded for when load-bearing). **Episode commits**: lesson-bearing → durable INFO named; ordinary per-turn → DEBUG. **Reflection conclusions** → one INFO per reflected turn (failed request → diagnosis → plan head → verify verdict) via `GhostReflect`.
- *Icon de-collision*: EVENT_BUS 📡→🔀, VECTOR_EMBED 🧬→🧮 (both wide-base; added to app.js `ICON_CLASS` + `?v=3.8→3.9` bump per convention); SKILL ACQUIRED 🎓→💡 (🎓 now reserved for real graduation in dream.py — its usage test still green); **all 18** "Verifier"-titled 💭 BRAIN_THINK icons → 🧪 VERIFIER_LAB via a context-aware transform (one icon = one subsystem; REFUTED no longer wears the thinking icon).
- *"Loop Breaker" split*: three distinct events → three distinct titles: **Failure Cap** (dispatch-pipeline fail cap), **Strike Cap** (turn-loop strike cap), **Think-Loop Halt** (thinking-loop force-final). The dispatch-extraction landmark test updated to the new marker; `test_agent_tool_limits` was coupled to different content (unaffected).
- Docs icon catalog updated. Full suite green; deployed.

**Logging work is CLOSED** — nothing deferred remains.

### 2026-07-24 — Outcome-gated learning loop: the FAILURE arm the retrieval-feedback loop never had

**Feature (research → build).** Surveyed the metacognition / self-improvement frontier against the current cognitive
stack; the one high-impact gap that honoured the grounded-or-off doctrine was **closing the memory learning loop with
real outcome labels**. The retrieval-feedback loop was **credit-only**: `record_helpful_retrieval` /
`credit_recent_retrievals` bump `helpful_retrievals` on success (relevance-gated), but **nothing ever debited a lesson
for being surfaced on a turn that FAILED**. So `hit_rate = helpful/retrievals` conflated "present on an uncredited
success" with "present on a failure", and a harmful lesson could ride along on failing turns invisibly — the
*experience-following* failure mode (retrieved memory reproduced whether appropriate or not; good and bad lessons
transfer with equal fidelity). The existing scrub (`retract_lessons_from_trajectory`) only removes lessons *authored in*
a failed turn, never lessons *retrieved into* one. And `judge_hydration_usefulness` credits via an **LLM judge** of
surface usefulness — the curation mode the 2026 literature shows silently fails.

**Mechanism.** Two new lesson arms `succeeded_retrievals` / `failed_retrievals` (schema + `_normalize_lesson`
back-fill, legacy-safe). New bulk one-write `SkillMemory.record_surfaced_outcomes(triggers, success)` (mirrors
`record_retrievals_bulk`). `compute_lesson_utility` folds in `outcome_mult = 0.4 + 0.75·out_rate` where
`out_rate = (succeeded+1)/(succeeded+failed+2)` — ×0.40 for present-only-on-failure (sinks below the prune cutoff),
×1.15 for present-only-on-pass. **The loop closes through the existing bounded prune** (`prune_low_utility`: ≤25%/pass,
verified pinned, ≥5 retrievals) — no new destructive path. Cold-start-neutral until `_OUTCOME_MIN_OBS=4` decisive
outcomes accrue (competence-Beta philosophy).

**Grounding — real outcome, not self-report.** Keyed off `resolve_turn_outcome` (passed/failed) — a structural
execution failure or a real verifier verdict — never the LLM hydration judge, never absence-of-failure. Only decisive
outcomes are recorded. Wiring (`Agent._record_lesson_outcomes`) runs at **both finalize paths** on both outcome classes
(outside the success-only credit gate). Decisive-at-finalize → record now; undecided (clean turn, async-critic verdict
pending — the prod case) → stash surfaced triggers by `trajectory_id` in a bounded map that
`_backfill_trajectory_outcome` drains via `_flush_stashed_lesson_outcome` when the late verdict lands. This gives the
late-verdict path a **second consumer** and captures the REFUTED signal a finalize-only hook would miss.

**Defaults / safety.** Recording always on (pure bookkeeping, zero behaviour change). The ranking/prune effect is
governed by **`GHOST_LESSON_OUTCOME_UTILITY`** (default on; `=0` → record-only, so the operator can watch the arms fill
before letting them prune) — see §3. Deliberately **on the retrieval/utility axis only**, NOT the smart-memory
write-admission threshold (`adaptive_threshold.py` stays inert — changing what gets *stored* is a separate
operator-visible retention decision, per its own docstring). Read-only skill proxies (`ReadOnlySkillMemory` + dream's
sim proxy) block the new mutator so idle sims never touch the prod playbook.

**Relation to #4/#27b.** This is the **live observational-utility instrument** for "does a learned lesson improve
outcomes?" — per-lesson success/failure co-occurrence on real traffic. It is NOT a revival of the CLOSED
synthetic-ablation earn-keep harness (§6 2026-07-23 later 3, do-not-resurface); it is the *different instrument*
(observational mediation on live trajectories) that closure explicitly left open. It does not by itself settle #4, but
it is the first grounded per-lesson outcome signal the corpus has ever carried.

**Verification.** New suite `tests/test_lesson_outcome_utility.py` (13 tests: schema, recorder, failure-arm demotion,
cold-start neutrality, kill switch, end-to-end prune of a present-on-failure lesson, verified-pin preserved). Touched-
surface regression: **877 passed / 1 skip** (skill/lesson/playbook/readonly/dream/prune) + **503 passed / 1 skip**
(finalize/verifier/late-verdict/calibration/trajectory). Docs: `docs/memory/skills.html` (schema + utility formula +
new *Outcome-gated learning loop* section + method table), `docs/memory/adaptive_threshold.html` (axis-distinction
note). Memory `[[outcome-gated-learning-loop]]`; relates to `[[built-but-unwired-loops]]`.

**LIVE-VALIDATED (2026-07-24, prod :8000, async-critic + `GHOST_LESSON_OUTCOME_UTILITY` unset→ON).** 4 driven turns:
(1) execution-failure turn → 5 surfaced code/error lessons `failed_retrievals 0→1` (synchronous decisive path);
(2–4) clean verified-CONFIRMED tasks → `succeeded_retrievals++`. The **streamed** path (call site hardcodes
`verifier_backfill=None`) recorded success — provable ONLY via `_flush_stashed_lesson_outcome`, so the stash→drain is
exercised and correct; every increment is exactly **+1 (no double-count)**. Flagship: *"When executing code, always
anticipate and handle errors"* reached `succ=3 fail=1` (present on 3 passes + 1 failure, each attributed to the right
arm; just crossed `_OUTCOME_MIN_OBS=4` so its `outcome_mult` is now live). Zero errors/tracebacks in
`_record_lesson_outcomes`/`record_surfaced_outcomes`/`_flush_stashed_lesson_outcome` across the session; RSS flat
(436→435 MB); playbook JSON intact (50 lessons, 12 now carry ticks). Prune effect not yet observed (runs idle/dream).

**Observability (added + verified after a launchd redeploy).** Per-turn recording → `logger.info`
(`lesson-outcome: N surfaced lesson(s) -> present-on-{FAILURE|success}`, + `… stashed … await late verdict`) in the
grep-log `$GHOST_HOME/system/ghost-agent.log` — the same sink as `Hydration tiers`, NOT the operator's pretty stream
`~/Data/AI/Logs/ghost-agent.log` (stdout/pretty, WARNING+). Actionable prune → `pretty_log("SKILL PRUNE", "… K
outcome-gated (present-on-failure): '<trigger>' (succ,fail)")` in the pretty stream (was generic "low-utility"). Matches
the file-only-vs-pretty convention (`[[logging-convention]]`, now documenting the two log destinations). +2 logging
tests; full suite **9031 passed / 13 skip**. Verified live post-redeploy: `13:09:50 … lesson-outcome: 5 surfaced
lesson(s) -> present-on-FAILURE` and a `present-on-success` line, both in the grep-log.

### 2026-07-23 (later 4) — #5 step 4a: client-SSE streamer extracted from handle_chat (LIVE-VALIDATED)

Continued the handle_chat decomposition (steps 1–3 already shipped). **Step 4a** extracts the client-facing
streaming branch (`if is_final_generation and stream_response:`, ~711 lines) into a new
`GhostAgent._stream_final_generation(self, ss: StreamState)`. Method: `symtable` capture analysis → the 2 nested
closures (`stream_wrapper` + `_stream_then_unregister`) close over exactly **26** handle_chat locals → new
read-only `StreamState` dataclass (beside TurnState/FinalizeState). Extraction shape: unpack `ss.*` → locals at
the method top, then the closure bodies moved **byte-for-byte** (uniform 16-space dedent, zero in-body edits), so
equivalence is by byte-identity. Done via a validated transform script (ast.parse + symtable zero-frees + dedent
gates) writing a candidate file, reviewed, then swapped in.

**Two bugs the transform introduced, both caught before trust:**
1. I wrote the method `async def` → handle_chat returned an un-awaited coroutine (`cannot unpack`). It's a plain
   `def` (the original branch had no top-level await — only lazy async generators — so it returns the tuple
   directly).
2. **The real one:** `_stream_owns_unregister = True` (the flag telling handle_chat's finally to DEFER the
   turn-unregister to the stream drain) was inside the moved region, so it set a method-local while
   handle_chat's frame stayed False → its finally unregistered the streaming turn prematurely (invisible /
   uncancellable mid-drain). Caught by `test_streaming_tail_cancellable`. Fix: set the flag in handle_chat's
   frame before the return. This is the WRITE-BACK case steps 1–2 flagged; my symtable pass only enumerated
   *reads* (free vars), not this cross-frame *write*.

**Collateral caught by the full suite:** 4 stale source-inspection tests (markers moved into the method:
loop-detector `tail=full_content[-400:]`, smart-mem `_is_int_req_m1`, work_log `_write_project_work_log_safe`) +
**9 `test_memory_store_durability.py` tests** that were actually left broken by the EARLIER journal overflow-spill
change (my narrower `-k` sweep hadn't matched that filename) — all updated to the lossless overflow model. Lesson:
run the FULL suite after a semantics change, not just a name-filtered slice.

Verification: **full suite 9016 passed / 13 skipped / 0 failed.** New guards `tests/test_stream_client_extraction.py`
(the load-bearing one: `_stream_final_generation` must have zero symtable free vars, so a future edit can't
reintroduce the mid-stream NameError). **LIVE:** operator restarted prod (which also fired the journal recovery —
"Recovered 1 in-flight journal item"); a `stream:true` probe streamed `chat.completion.chunk` deltas → clean
`[DONE]`, zero errors in ghost-agent.err. **Next: 4b** (Region B internal consumer, ~1,170 lines, turn-loop tail +
the `_emit_thinking`/`_flush_thinking` nonlocals — its own method with a repack, like step 2). Plan:
`.claude/plans/spicy-foraging-pudding.md`; memory `[[agent-py-decomposition]]`.

### 2026-07-23 (later 3) — Earn-your-keep / synthetic-ablation route CLOSED as inconclusive (operator decision)

Operator's call after the full arc: **drop the synthetic-battery route entirely and declare the
self-measuring→self-pruning experiment inconclusive for this model.** The premise was that auto-graded
deterministic tasks could measure whether each cognitive subsystem earns its keep. Two independent batteries
proved they can't on this uncontended 35B — Track A puzzles ceilinged (full ≈ thin ~98% once scored fairly),
Track B4 grounded DOING tasks ceilinged (32/35 at 3/3, the 3 "survivors" pure 300s-timeout flakes). The
statistical auto-prune rule the harness is built around will therefore **never fire on this instrument**.

**Outcome: NOTHING pruned. Prod config unchanged — every subsystem stays on.** This is a deliberate, honest
non-result, not a deferral. What the arc DID leave, all standing on their own merit:
- **Findings:** the per-turn cognitive stack, in aggregate, is *neutral* (neither helps nor hurts) on every
  measurable task; the **verifier is exonerated** (confirms 100%, never false-refutes — it had been the
  prime suspect). Per-subsystem resolution was never achievable (LOO deltas all noise under the ceiling).
- **Real fixes that outlive the experiment:** the `final_number_is` scoring artifact (ablation_hard_tasks),
  the **journal overflow-spill data-loss fix** (memory/journal.py — [[journal-overflow-spill]]), the
  `--no-dream` / `--no-self-play` idle-loop gates (useful ablation levers regardless).

**Disposition of the infra (NOT deleted):** `scripts/earn_keep.py`, `ablation_*`, `core/prune_overrides.py`,
the idle-loop toggles, and the ledgers/pilot artifacts stay in the tree as **dormant** infrastructure. They
would only be revived if the *instrument* changes — the two live-but-unfunded ideas being observational
mediation on real production trajectories, or ablating against a deliberately degraded/contended model. Until
then this is a closed chapter; do not re-propose running more synthetic batteries. Records updated: §4
"(1) Blocked on operator action" (rewritten to CLOSED), [[earn-keep-harness]] memory.

### 2026-07-23 (later 2) — Memory journal made LOSSLESS (overflow spill); Track-B synthetic battery is a dead end

**Track-B pilot verdict (calibration gate):** ran `ablation_trackb4.py --pilot` (35 grounded DOING candidates ×3
passes). **32/35 tasks ceiling at 3/3**; the 3 "survivors" are pure **timeout flakes** — every failure is the
one pass that hit the 300s timeout with an empty artifact; the model solves all 35 when given time. So there is
**no calibrated Track-B battery either** — the same ceiling as Track A (2026-07-23 earlier), now on grounded tasks.
Strategic read logged: two independent auto-graded deterministic batteries both ceiling on this uncontended 35B →
the synthetic-battery instrument can't measure this machinery on this model; hand-authoring a third harder battery
is a treadmill. Real options for Track B: (A) observational mediation on real trajectories [recommended], (B)
degrade the model to open headroom, (C) accept the null and make the costed idle loops opt-in. Deferred to operator.

**The journal-drain bug the pilot exposed — FIXED.** During the back-to-back pilot the operator saw repeated
`Memory journal is full (capacity 50): discarded 1 oldest ... The drain (~2 min idle) is not keeping up`. Root
cause (`memory/journal.py`): the smart_memory buffer drained ONLY on idle (each consolidation ~90s LLM, must not
compete with the user), so under sustained load with no idle gap it filled and **silently dropped the OLDEST**
consolidations. Two harms: (1) it starves the dream arm by construction (dream counts the `type:"auto"` fragments
smart_memory writes — dropped during back-to-back seeding); (2) genuine production data-loss under bursty load.

Fix — made the journal **lossless by construction** (not just a bigger buffer):
- Logical queue is now `overflow + hot` (oldest→newest). `append()` past the cap SPILLS the oldest surplus to a
  new `memory_journal.overflow.json` (drained oldest-first) instead of dropping. `load()` still returns hot only;
  `pop_all()`/`drain()` take overflow+hot. Hot cap raised 50→256.
- `push_front()` (drain requeue) folds to the overflow HEAD (drained first); deleted the old capacity-bounded
  `_merge_front` that discarded surplus. `recover_inflight()` folds a crash-interrupted batch to the overflow head
  too — crash-recovery is now lossless even when the batch exceeds the hot cap.
- New `pending_count()` (hot + overflow). **Critical wiring:** the two "is there work?" gates —
  `agent.py` biological phase-1 and `tools/memory.py` self-play inter-cycle drain — now use `pending_count()`, not
  `len(load())`; otherwise overflow-only work (a spilled burst, or a requeue) would sit undrained forever.
- Dropped an early over-design: a hard overflow ceiling whose `.spill-{int(time.time())}` sidecar name COLLIDES
  within one second under a tight loop → would overwrite/lose. Removed it; overflow is unbounded-but-lossless
  (small text, any idle gap drains it).

Verified end-to-end: 300 back-to-back appends (old code would drop 250) → 300/300 preserved, drained FIFO, overflow
file self-clears. Tests: `tests/test_journal.py` (+burst/recovery/push-front-overflow), updated
`test_memory_audit_fixes.py` / `test_deep_audit_fixes.py` / `test_smart_memory_requeue.py` /
`test_self_play_loop_and_lessons.py` / watchdog+bio-tick mocks (stub `pending_count`). 782-test memory/dream/
self-play sweep green. Docs: `docs/memory/journal.html`.

### 2026-07-23 (later) — Track B (Phase 2) BEGUN: idle-loop disable gates + pilot verified ready

Chose the strategic fork after the Track-A ceiling (see prior entry): **pivot earn-keep to Track B** — adjudicate
the cross-session idle loops (dream / self-play / reflection), the one genuinely open question and where an effect
is already known to exist (memory 98% vs 0%). Phase-2 order is pinned by the plan AND the fresh Track-A lesson:
**calibrate the probe battery before building measurement.**

**De-risked the calibration gate (operator's multi-hour job).** The B4 harness (`ablation_trackb4.py`) is verified
runnable without the model: 35 probe candidates / 8 clusters / 8 seeding tasks all construct; **103 battery
self-consistency tests pass**; pilot imports + arm-flag construction clean. Operator command handed off:
`PYTHONPATH=src python scripts/ablation_trackb4.py --pilot --report-dir ablation_out/b4-pilot` (prod stopped, 35B up).

**Built the two missing idle-loop disable gates** (a real per-loop LOO needs them; only `--no-reflection` existed):
- `--no-dream` — gates the Deep REM Dream phase (biological-watchdog phase 2) at `agent.py`; leaves reflection +
  self-play intact.
- `--no-self-play` — gates the Synthetic Self-Play phase (phase 3, `agent.py`) — both fresh self-play AND the
  counterfactual-replay slot. **Distinct from `--no-frontier-selfplay`**, which only changes cluster *selection*,
  not whether self-play fires (this was the trap — the plan assumed frontier-selfplay was the self-play toggle).
- Catalog (`core/prune_overrides.py`): added `dream` (arm `full_no_dream`) and `self_play` (arm `full_no_selfplay`)
  as Track-B, **costed** (real idle LLM work), non-protected subsystems, alongside `reflection`.

**Gotcha handled:** the gates read `getattr(ctx.args, "no_dream", False) is not True` — NOT a plain truthiness check
— because ~10 watchdog test files mock `ctx.args` as a `MagicMock`, whose auto-vivified `.no_dream` child is truthy
and would spuriously disable the phase. `is not True` matches only a real argparse store_true. Verified: 778 idle-loop
tests pass, flags parse (default OFF, disable when passed).

Tests: `tests/test_biological_watchdog.py` (+3: dream-off skip, self-play-off skip, **isolation** — `--no-dream`
leaves self-play firing); `tests/test_earn_keep.py` (+3: idle loops catalogued Track-B/costed, arms distinct,
arg-apply flips `no_dream`/`no_self_play`). Docs: `docs/cli_reference.html` (both flags), `scripts/ABLATION.md`
(Phase-2 progress table + the remaining 3 steps).

**Next (gated, in order):** pilot → collective `treatment` vs `control` (existing harness, no new toggles) → only if
non-null, wire `earn_keep run --track B` to boot these arms through the trackb4 seed→idle→probe protocol and fold
probe outcomes into the ledger. Building the `--track B` orchestration BEFORE the collective result would repeat the
Track-A over-build mistake, so it waits.

### 2026-07-23 — earn-keep Track A: the "full stack loses to thin" headline was a SCORING ARTIFACT (verifier exonerated)

Ran earn-keep Track A three times (`full` + 8 leave-one-out arms + `thin`, 3×16 tasks/arm/run = 432
records/run, idle-quiesced, clean). Headline across all 3 runs: `full` 83.3% vs `thin` 86.8%, gap −3.5pp —
i.e. the full per-turn cognitive stack appeared to *underperform* stripped, reproducing the 2026-06-28
suspicion. The per-arm split flagged the **verifier** as the biggest apparent loser (`full_no_verifier`
scored the *highest*, 87.5%), so the operator asked to dig into it before pruning anything.

**The dig reversed the conclusion.** The verifier CONFIRMED 100% on these tasks — it never refuted a correct
answer; my initial "false-refute" hypothesis was wrong. The harm was entirely a **validator artifact**:
`final_number_is(n)` scored the *last numeric token* in the reply, and a correct, verified answer that shows
its work ("the father is 36 … in 12 years 48 = 2×24 ✓") ends on **24** → marked wrong. Verbose arms
(verifier, deep-reason, metacog) emit more verification prose than `thin`, so the rule docked them for
*being verbose*. Proof by cross-tab over the ledger:

| bucket | full | thin | gap |
|---|---|---|---|
| 12 lenient tasks (`contains_number`) | 98.1% | 99.1% | −0.9% (tie, at ceiling) |
| 4 strict tasks (`final_number_is`) | 38.9% | 50.0% | **−11.1%** |

The entire −3.5pp headline lived in the 4 strict-scored tasks; on fairly-scored tasks the stack *ties* thin
(and those are ceilinged at 98–99%, so they can't discriminate anyway). **Pruning metacog/deep_reason on
this data would have pruned on an artifact.** Auto-prune had (correctly) not fired — every verdict was still
`insufficient`/`keep` because the CI-upper-<+2pp gate held.

**Fixes (this session).**
- `scripts/ablation_hard_tasks.py` — rewrote scoring around a canonical `ANSWER: <value>` line. Every prompt
  is auto-suffixed with the directive; new validators `answer_int` / `answer_num` read the **last `ANSWER:`
  line** (lenient standalone-token fallback if the marker is missing — so a forgotten marker isn't penalised,
  but trailing prose can't hijack the score). Removed `final_number_is` / `contains_number` /
  `contains_any_num`.
- Added **8 intuition-trap reasoning tasks** (`bat_ball`, `algae_quarter`, `printers_pages`, `avg_speed`,
  `pct_updown`, `overlap_sets`, `compound_discount`, `clock_angle_315`) to add failure-frontier headroom
  where reasoning/verification *can* pay off; parameters shifted off textbook values so a recalled trick
  doesn't win. Suite 16 → 24 tasks.
- `tests/test_ablation_hard_tasks.py` — 51 tests: every integer answer recomputed independently (incl. the 8
  traps), decimal-form acceptance, and the regression guard `test_answer_line_beats_trailing_verification_prose`
  (the exact reply that used to fail must now pass) + `test_last_answer_line_wins`.
- Archived the contaminated ledger → `ablation_out/earn_keep/results.pre-answerline.jsonl` (the 3 runs were
  scored under the biased rule; pooling them with fixed runs would re-corrupt `report`). `report` on the
  fresh slate is clean (0 runs).
- Docs: `scripts/ABLATION.md` — new "Scoring — the `ANSWER:` line (and why last-number was a trap)" section
  + a "Ledger hygiene" callout.

**Lesson (generalises):** an ablation's *validator* is part of its measurement. A validator that correlates
with an arm's **style** (here: verbosity) rather than its **correctness** will invent differences that aren't
there. Score the declared answer, not the last digit.

**Still open / next:** the difficulty of the new trap tasks is UNCALIBRATED (couldn't run the 35B this
session — prod stopped for the operator). Whether they actually break the ceiling and land in a
discriminating band is confirmed by the next run's pass rates; tune from there. Then re-run Track A 3× on the
fixed battery before trusting any keep/prune verdict. The `--track B` idle-loop adjudication (Phase 2)
remains the bigger unanswered prize.

### 2026-07-22 (later 5) — STRATEGIC PIVOT: the earn-your-keep self-measuring/self-pruning harness (Phase 1)

After closing the bug-hunt cycle, the agreed next big thing is to stop flying on faith: make the agent
**self-measuring, then self-pruning**. Every cognitive subsystem must prove it earns its keep or get
pruned. This is Phase 1 (Track A, in-session) — an ORCHESTRATION build on the ~70% existing ablation
machinery, not greenfield. Plan: `.claude/plans/spicy-foraging-pudding.md`. Operator decisions: standing
infra + Track A first; **on-demand** runs (no nightly prod downtime); **auto-prune** on a sustained
verdict (reversible, loud). Suite green (8957 passed; the 1 "fail" is a pre-existing TERM-brittle CLI
test — passes with a normal `TERM`). DEPLOYED (both prod changes are validated no-ops until a prune
exists).

**What shipped:**
- `scripts/earn_keep.py` — the standing harness. `run` boots the leave-one-out config matrix (`full` +
  one `full_no_<x>` per subsystem + `thin`) via the paired driver, fires `ablation_hard_tasks` at every
  arm back-to-back, APPENDS raw per-(arm,repeat,task) pass/fail to a durable ledger
  (`ablation_out/earn_keep/results.jsonl` — the trending substrate), then re-attributes over the WHOLE
  ledger and auto-prunes. `report` renders the ranked keep/prune table (Δ-help, 90% bootstrap CI, latency
  cost, verdict). The marginal contribution of X = the paired `full` vs `full_no_x` delta.
- `src/ghost_agent/core/prune_overrides.py` — the single-source-of-truth catalog (subsystem ↔ ablation
  arm ↔ toggle) + the prod-apply (flip arg attrs / set env) + protected-set refusal + defensive load
  (absent/malformed → `{}`, never raises).
- `src/ghost_agent/main.py` — reads `$GHOST_HOME/system/earn_keep/pruned.json` at boot: env-kind prunes
  applied BEFORE `core.agent` import (its toggle constants read env at import), arg-kind after
  `parse_args()`, each logged loudly. Clean no-op when the file is absent (verified: a malformed file
  also boots clean — the load-bearing safety invariant).
- `scripts/ablation_paired.py` — extended `CONFIG_FLAGS` to the full LOO (added `full_no_verifier`,
  `_selfmodel`, `_workspacemodel`, `_hypothesis`) via a `_full_minus` diff helper + a per-arm `CONFIG_ENV`;
  `scripts/ablation_eval.py::_boot` gained a backward-compatible `extra_env`; `core/agent.py`
  `_HYPOTHESIS_GROUNDING_ENABLED` now reads `GHOST_HYPOTHESIS_GROUNDING` (the only new agent toggle).
- **Pre-registered auto-prune rule** (do not move post-hoc): prune a non-protected subsystem only when,
  across ≥3 runs / ≥60 matched pairs, Δ ≤ 0 AND 90% CI upper < +2pp AND it carries a compute cost.
  **Protected (measured, never auto-pruned):** memory (Track-B-proven 98% vs 0%), verifier (correctness).
  Every prune is reversible (delete the pruned.json entry), auditable, loud.

**Operability (added same session):** `run` is a SINGLE resumable entrypoint. Each run gets
`ablation_out/earn_keep/run-<id>/` with a unified `progress.log` (tees to stdout + file, absorbs the
boot/teardown chatter — `tail -f` mirrors the live view), per-arm agent logs (`full.log`, `thin.log`,
…), and `checkpoint.jsonl`+`manifest.json`. Checkpointing is at (repeat,task)-GROUP granularity — all
arms fire back-to-back (paired property survives a resume) and a group is written atomically, so a
kill mid-group redoes only that group. `run --resume` (or `--run-id`) skips done groups; a run's data
folds into `results.jsonl` ONLY on full completion (no partial run pollutes the trend); a fresh run
while one is incomplete is refused (→ `--resume`/`--force-new`). The prod-down preflight runs BEFORE
any run-dir is created, so a refused run leaves no resumable stub. The run prints the `tail -f`
commands at startup.

Tests: `tests/test_earn_keep.py` (35, pure — attribution math, verdict every branch, prune I/O +
prod-apply + protected refusal, PLUS the resume machinery: group-done detection, atomic checkpoint,
dedup-fold, manifest/find-resumable, and the real `_measure_resumable` loop with a fake runner proving
resume skips done groups — no live boot). Docs: `scripts/ABLATION.md` §Earn-your-keep.

**Deliverable ready for the operator:** `PYTHONPATH=src GHOST_HOME=<live> python scripts/earn_keep.py
run --track A --repeats 3` (prod stopped) produces the first attribution table; repeated runs trend it;
sustained losers auto-prune. **Phase 2 (needs an operator evening):** calibrate the B4 battery
(`ablation_trackb4.py --pilot`) + wire the Track-B idle-loop LOO (dream/self-play/reflection) into
`run --track B` — adjudicating the idle loops, the one genuinely open question.


### 2026-07-22 (later 4) — SANDBOX/EXECUTION + external-infra review (4 agents) + main-host FIXES

First dedicated hunt of the sandbox/execution + external-services layer (docker.py, services.py,
execute.py's contract, the external GPU servers, the uConsole/Slack clients) — the containment
boundary every code execution and build passes through, never dedicated-hunted this cycle. 4 review
agents. Prod is macOS/**bridge** (verified), so the LAN-exposure findings are Linux-only. Fixed the
main-host cluster (deployable via plain-kill); the node-deployed items (voice server, uConsole/Slack
clients) were **skipped per operator — not in use**.

Coordinator owned `sandbox/docker.py` + `tools/execute.py` (the coupled pair); the services.py review
agent was resumed to fix its own findings in `sandbox/services.py`. Suite green.

**docker.py (mine):**
- **Client-side exec deadline (prod-critical).** docker-py's exec socket read blocks in `poll.poll()`
  with no timeout (verified in 7.1.0), so a wedged daemon hung the worker thread forever — and the
  provision execs hold `self._lock`, wedging EVERY other turn's execute with zero logs. All 18
  `exec_run` sites now go through `_exec_run(cmd, deadline_s)` (daemon thread + `join` deadline → raise
  `SandboxDaemonTimeout`, releasing the lock). Main command uses `timeout+60`; others use a generous
  `GHOST_EXEC_DAEMON_DEADLINE` (1200s) that only fires on a true wedge.
- **Infra failures marked** `[SANDBOX INFRA ERROR — not your code]` instead of a bare exit 1 (the model
  was debugging its own code on a sandbox fault); the marker also keeps execute.py's heal from firing.
- **Resume stopped containers** (`_try_resume_stopped`) instead of destroy+reprovision — an RSS restart
  no longer nukes in-sandbox services + runtime state (the `close(remove=False)` "fast resume" the
  docstring promised never existed).
- **Adopted-container published ports** derived from live `PortBindings` (`_derive_published_ports`) on
  the adopt + 409 + port-conflict-retry paths — the remote hint can't point the operator at a foreign
  process's port. Added `binds_host_netns()` for the Linux HOST-export follow-up.
- **Spill-log counter** seeded past existing `run_N.log` (no clobber after a plain-kill deploy).

**execute.py (mine):** the cwd-heal no longer re-runs a command that already executed — timeout kills
(124/137/143) excluded from the heal trigger, and `_looks_like_file_not_found` returns False on output
containing a `Traceback` (a traceback = the script ran → a "no such file" is a runtime data-file error
after side effects, not a wrong-cwd miss). One stale test (`test_root_retry_keeps_real_fnf_traceback…`)
updated: it pinned the old "re-run then reject" behavior; the fix is strictly better (no re-run, same
user-visible result).

**services.py (review agent):** port-reclaim ownership check (never TERM/KILL a process another
registry entry owns; `start()` refuses a port a live service holds); container-generation stamp (a
recycled pid in a new generation reads DEAD → kills phantom-RUNNING and makes "listening ✓"
trustworthy); atomic restart preserving the registration on relaunch failure (`_lock`→RLock); dead
`_reap_dead` removed.

Tests: `test_docker_review_fixes.py` (10), `test_execute_heal_timeout_guard.py` (7),
`test_sandbox_services_review_fixes.py` (14). Docs: sandbox/docker.html, sandbox/services.html,
tools/execute.html.

**LOGGED, not fixed (in §4B):** node-deployed items — voice server (Orin) no-auth CRIT + sync-on-loop +
unbounded upload; uConsole client SSE token-loss + camera/QTimer leak + send-race + unbounded history;
Slack thread-boundary admits any bot + `file_share` DMs dropped (owner-lock itself holds). Linux-only:
`HOST=0.0.0.0` host-mode exposure (accessor added, services-side export pending), exec-user
provisioning, egress-guard loopback bypass. Deferred: readiness false-negative removes a healthy
container (self-heals); the full infra exit-code sentinel (light `[SANDBOX INFRA ERROR]` marker done).

### 2026-07-22 (later 3) — MEMORY-SUBSTRATE review + FIXES: the highest-damage cohort yet (multiple CRITs with LIVE data loss)

**DEPLOYED + DATA-REPAIRED + LIVE-VERIFIED 2026-07-22.** Clean suite 8901 passed / 0 failed (the 2
`test_thinking_loop_guards::TestAtomicPrint` "failures" are a shell `FORCE_COLOR=3` artifact — pass
with it cleared; logging.py untouched). Operator restarted; 20 wrongly-expired graph OWNS/IS rows
restored via `repair_owns.py --apply` (backup `knowledge_graph.db.pre_owns_repair_1784724404.bak`);
collapsed `weights.json` removed (bus → healthy priors, dream refit regenerates). LIVE VERIFICATION:
asked the running agent "what vehicles do I own?" (no tools) → answered "BMW 118i, Ducati Streetfighter
V4s, Sym scooter" — all three were expired by the bug and now surface correctly. **DOCS DONE** (4
parallel agents; all 11 memory HTML docs — vector/readonly/graph/episodes/scratchpad/bus/journal/
profile/competence/contradiction_log/adaptive_threshold — updated + verified vs source + tag-balanced;
graph.html's flagged-stale functional-predicate section corrected). Backups retained: graph pre-repair
DB + collapsed weights, both in the job scratch dir.


First dedicated hunt of the retrieval/consolidation substrate (vector/graph/episodes/bus/rrf/journal +
small stores) — the cognitive core every turn depends on, never swept this cycle. Chosen precisely
because memory bugs are SILENT (a retrieval or consolidation corruption doesn't throw, it quietly
poisons future turns), so log-watching can't surface them. 5 parallel review agents (each validating
against the LIVE production stores read-only) → 5 parallel one-file-per-owner fix agents + coordinator
(owned vector.py/readonly.py/agent.py). This cohort found the most severe issues of any subsystem so
far, several with damage that had ALREADY happened on live data.

**Data loss confirmed on the live stores (verified by the coordinator):**
- **CRIT — graph `OWNS`/`IS` wrongly in `_FUNCTIONAL_PREDICATES`** (both multi-valued): every new
  `user OWNS X` extraction expired ALL previous ones. 19 of the 21 expired rows in the whole graph
  were this bug — the agent had "forgotten" the operator owns a BMW 118i, a Ducati Streetfighter V4S,
  **evolmonkey (his own company)**, webos, piggybag, nova, chess coach. Fix (graph agent): removed
  OWNS/IS; added the genuinely-functional operational predicates (HAS_STATUS/STATUS/HAS_PID) that were
  MISSING and accumulating contradictions (`chess-v4 HAS_STATUS dead` AND `running` both current);
  instrumented every expiry with a WARNING. **DATA REPAIR: 20 wrongly-expired rows restored via
  scripts repair (backup-first, dry-run verified), operator-authorized.**
- **CRIT — scratchpad wiped to 2 sentinel rows**: any non-owning conversation's request start deleted
  every key incl. in-flight swarm results (`_hydrate_scratchpad` had no scope primitive). Fix
  (scratchpad agent): a per-entry namespace tag + `clear_namespace` so parking clears only the foreign
  project's scope and NEVER a background-job's output_key.
- **CRIT — RRF intent-weight matrix collapsed into noise**: `fit_intent_weights` calibrated "rate 0.5
  → base weight" but the real judged-used rate is ~0.14, so every well-sampled cell was crushed to
  WEIGHT_MIN — live `factual/graph` fitted 2.0 → 0.1 (a 6.6× INVERSION), intent routing silently
  inert. Fix (bus/RRF agent): relative-multiplicative fit `weight = prior*clamp(lift^(GAMMA*shrink))`
  calibrated on the OWN base rate, evidence-shrinkage toward the prior, hard `MAX_DEVIATION=2.0` band
  (can re-order mildly, never invert); turn-share normalisation for the items-per-turn confound.
  Coordinator-verified the arithmetic (collapsed cell → 1.84, near its 2.0 prior). **weights.json
  regenerates on the next dream refit (anchors on defaults); deleting it heals immediately.**

**Silent degradation the coordinator verified + fixed directly (vector.py/readonly.py):**
- **HIGH — ambient hydration was DARK since the 2026-07-13 manual ingest.** 96.8% of the store is
  ingested-document chunks; documents get a 1.25 threshold (2×) + p_score=-5 and lower combined_score
  wins, so a barely-related doc (dist 1.0 → -0.5) beat a strong auto memory (dist 0.30 → +0.60).
  MEASURED on a live-store copy: the 30-candidate ambient pool was **30/30 document chunks** (0
  episodes/identity/skills). Fix: `_search_selection` excludes `type=document` (doc QA has its own
  scoped `search_document`); after the fix the same query yields 18 episodes + 6 identity + 6 skills.
- **CRIT (latent) — an embedding-function conflict silently DELETED the whole store.** The
  `delete_collection`+recreate recovery sits in an `except` matching "already exists"/"Embedding
  function conflict" and is reached BEFORE the fingerprint guard (which is after the raising
  `get_or_create_collection` in the same try). A chromadb upgrade or EF-class change would have wiped
  all 7,368 fragments. Fix: count first, refuse to reset a populated collection (sys.exit + re-embed
  instruction), keep the auto-reset only for an empty collection.
- HIGH — `search_advanced` unconditionally bumped retrieval stats on every raw hit (the episodic tier
  credited ~5-8 rows/hydration never shown to the model, poisoning prune-survival + decay). Fix: added
  `record_retrievals=False` + a `where=` scope; the read-only façade forces it off. HIGH — the
  read-only façade leaked `_update_library_index` (a self-play ingest permanently poisoned the
  operator's library index) and `search_advanced` (a delegated sub-agent mutated operator memory on
  every recall) — both now blocked/proxied. MED — `smart_update` denylist wasn't the complement of
  `_PRUNABLE_TYPES`, so an auto-extraction could delete a user `manual` memory or a dream synthesis;
  now same-type-only. MED — `add()` id-exists early return never refreshed twin metadata, so
  `retract_lessons_from_trajectory` silently missed and a discredited lesson stayed retrievable; now
  refreshes. Fragment-correction full scans (materialised 7k doc chunks/call) pushed into the query.

**Also fixed (episodes/consolidation agents):** action truncation kept the HEAD and dropped the
resolution (now head+tail, warns); `search_recoveries` was structurally dead (0/145 live — no writer
sets `lesson`; now derives the recovery signal from the live schema); float-epoch episode timestamps →
ISO; three small stores (competence/contradiction/adaptive_threshold) treated a corrupt/unreadable
file as EMPTY and overwrote it (total silent wipe) — now sidecar-on-corrupt + refuse-write-on-degraded
like journal/profile; journal `pop_all` gained crash-safe in-flight staging + recovery (a deploy-kill
mid-drain lost up to 50 items); `is_upstream_transient` widened for the plain RuntimeError llm.py
raises; profile singular relationship/possession keys REPLACE not accumulate; notes.info bounded.
**COORDINATOR SELF-CATCH: my own 2026-07-22 post-mortem requeue fix was INERT** —
`_RetryableConsolidation` wasn't imported in `_execute_post_mortem`'s scope (raised NameError) and even
fixed would have been swallowed by the function's own `except Exception`; now function-scope import +
a re-raise clause guarding the broad handler, runtime-verified to propagate.

Cross-file wiring the coordinator added to activate the bus fixes: agent.py passes
`raw_user_text=last_user_content` to `hydrate_context` (classify intent on the user's message, not the
expanded query); dream.py forwards the `turn` id to `fit_intent_weights` (activates turn-share
normalisation). Two stale pins updated for intentional contract changes (RRF all-failure now keeps the
prior not WEIGHT_MIN; graph temporal test IS→LIVES_IN). Tests: 10 new/updated files across the cohort.
STILL DEFERRED (logged, not shipped): episodes `_scoped_episode_hits` could use the new
`search_advanced(where=)` instead of the `.collection` handle (cleaner, coupling); a boot reconcile of
episode rows missing a vector twin; competence per-tool write debounce.

### 2026-07-22 (later 2) — the final two LLM-stack items: KV-pin stable-block + dream containment heal-open

The two items deferred from the "proceed with all" pass (below), both in `core/agent.py`, done
carefully with the risk they warranted. Full suite green (**8715 passed / 13 skipped**). NOT yet
deployed at time of writing.

1. **KV-pin stable-block instability (HIGH perf).** Under `GHOST_PIN_TOOL_SCHEMAS=1` (prod), the
   "stable" injection pinned to the first user message held two per-turn-varying pieces that busted the
   KV prefix: (a) the skill playbook (its lookup keys off the planner's per-turn `required_tool` +
   `thought_content`, so a shifted retrieval changed the pinned block), and (b) the final-gen turn
   flipped `tool_header_block` to the slim header at the very front — on the turn carrying full history,
   forcing a whole-history re-prefill. Fix, gated on `_pin_stable` so the UNPINNED path is byte-for-byte
   unchanged (the layout tests are all pin-OFF and stay green): under pin, `_stable_injection` excludes
   `fetched_playbook` (prepended to the volatile `dynamic_state` — it's genuinely per-turn-relevant) and
   keeps the byte-stable header on final-gen turns, routing the "answer directly, no `<tool_call>`"
   directive to `dynamic_state` (native/legacy `tools` are suppressed downstream regardless). `_pin_stable`
   is hoisted above the header assembly. Proof chain: `test_compose_injection.py::test_pinned_first_message_
   identical_across_turns` already proves identical stable string → byte-identical pinned message;
   `test_kv_pin_stable_prefix.py` (new, 4) proves the stable string no longer holds the per-turn playbook/
   directive under pin and the unpinned composition is untouched. Verify live via the `Prefill Cache h=`
   line (now stable across a planned request's turns including the final answer).
2. **Dream temp-agent containment heals open (MED).** `_rebuild_available_tools` (heals the dispatch
   dict after a hallucinated-name miss) re-narrowed only via `_subagent_allowed_tools`, which dream's
   self-play temp agent never sets — it contains by SETTING `disabled_tools` + popping `available_tools`.
   So a dispatch miss healed the popped tools back, and an aliased variant (`"websearch"`→`"web_search"`)
   could reach the network-egress tools self-play disables. Fix: the rebuild now also drops
   `self.disabled_tools`; since dispatch requires `fname in available_tools` and canonicalisation only
   returns names already in it, a disabled tool can't fire post-rebuild. Hardens the sub-agent path too.
   Tests: `test_subagent_containment.py::TestRebuild` (+2: dream case, allowlist∩disabled). Docs:
   `agent.html` (context-compaction §), `delegation.html` (containment §).

### 2026-07-22 (later) — LLM-stack review FIXES: contention theme + data-integrity + ~20 items across 13 files

"Proceed with all" pass over the 2026-07-22 LLM-stack review catalogue (§4B). Coordinator owned the
two shared hot files (`core/llm.py`, `core/agent.py`); 10 parallel one-file-per-owner agents fixed the
disjoint files. Full suite green (**8713 passed / 13 skipped**; 2 pre-existing source-pin/behavior
tests updated for the intentional swarm-raise + memory-timeout changes). NOT yet deployed at time of
writing — plain-kill to deploy.

**The dominant theme — main-slot contention — is now closed at the consumers.** `off_main_only=True`
threaded through every background LLM consumer so a worker hiccup degrades instead of dogpiling the
single main slot: dream/self-play (8 calls, `off_main_only=is_background` to exempt user-triggered
self-play), verifier last-resort fallback (bounded + `is_background` only when no user request live),
project-research idle summariser (background auto-detected from `foreground_requests`), main.py
selfhood/workspace critique closures. Plus `warm_up_workers` (concurrent `gather` + `off_main_only` so
a dead boot node can't burn/evict the main prefix), `targets_main_node` gained the coding pool, and
bounded timeouts on the untimed background calls (smart-memory extract 90s, post-mortem 90s, dream
challenge-gen/repairs 180s) so a wedged-but-connected node can't pin the journal drain 20 min.

**Data integrity:** the mid-stream fail-open (H, `core/agent.py`) — an upstream abort frame
(`data:{"error"}`, no `choices`) was silently dropped and the truncated reply finalized as complete
and fed to the verifier/memory. Now detected (`stream_errored`) and folded into `_truncated_text_turn`
so it triggers the continuation path; streamed-final path logs the abort. Circuit breaker no longer
counts HTTP 4xx (caller fault) as node faults (`_is_node_fault`, all 5 pool branches). Retry adds
`ConnectTimeout`/`PoolTimeout` (never-sent → idempotent); `ReadTimeout` still excluded. Post-mortem
transient failures now re-queue (the 2026-07-09 requeue fix had covered only smart-memory).

**Also fixed:** router boot landmine (a failed checkpoint load no longer kills the router + all
retrains — load wrapped in its own try, `_router_checkpoint_path` preserved so bootstrap overwrites the
bad file; `router/model.py` load raises clean ValueErrors — **do this before any router schema bump**);
`jobs(collect)` read-marking (was re-dumping ~400KB every call); swarm `await_results` deadline (240s,
partial return, workers not cancelled) + failed swarm work raises → FAILED job (was success-shaped
`[done]`); subagent containment fail-CLOSED (was fail-open on a registry-import throw); `delegate_to_swarm`
already gated (earlier today); `llm_recording` cross-restart session-id + torn-line tolerance + image
elision; `tool_grammar` sval trailing-partial-prefix + collision-free rule names + literal escaping;
`_aa_code_gen` max_tokens 1024→4096; `/api/health` now surfaces `node_health` (breaker state).

Tests: `test_llm_contention_fixes.py`, `test_agent_stream_bg_fixes.py`, `test_dream_offmain_contention.py`,
`test_verifier_offmain.py`, `test_router_boot_resilience.py`, `test_project_research_offmain.py`,
`test_jobs_collect_readmark.py`, `test_swarm_tool.py` (appended), `test_subagent_failclosed.py`,
`test_llm_recording.py` (appended), `test_tool_grammar.py` (appended), `test_health_node_status.py`.
Docs: llm/agent/dream/verifier/project_research/delegation/api-routes HTML updated.

**The two originally-deferred items — FIXED 2026-07-22 (later 2), see the "final two LLM-stack items"
entry below.** KV-pin stable-block instability + dream temp-agent containment heal-open both landed
with tests (`test_kv_pin_stable_prefix.py`, `test_subagent_containment.py::TestRebuild`), suite green.

**STILL DEFERRED (low-value / needs live infra):**
- **Streaming recording hook (dev feature).** `GHOST_LLM_RECORD` still misses streamed (all main-model)
  turns; adds buffering to the hot path. Low value (recording off in prod).
- **IMPROVEMENTS left:** node-payload serializer unify (`llm.py`, 80-line refactor, low value);
  verifier stage-1/2 shared prefix-cache block (needs live `verify_bench`); fallback-hint merge
  (`fallback_chains.py`+`tool_failure.py`, touches agent.py call sites).

### 2026-07-22 — LLM/routing/delegation-stack review (6 agents) + delegation/swarm cluster FIXES

First dedicated sweep of the inference layer every subsystem rides through — `core/llm.py`
(request-construction + failover), the `router/` package, `llm_recording.py`/`tool_grammar.py`, the
delegation stack (subagent/delegate/swarm/fallback_chains/qwen_bridge), and the consumer side
(verifier/dream/journal/advancer). Chosen because §5B/§5C and the 2026-07-20 reviews all covered
project/turn-loop/correction, never this layer — and the live post-deploy logs pointed here
(`delegate_to_swarm -> not configured` burning a strike, native tool_call repair, a verifier LATE
REFUTED). Six parallel review agents; every finding required file:line grounding + a concrete failure
scenario. Full catalogue logged in §4B "2026-07-22 LLM-stack review". **The dominant theme** (4+
agents converging): background/foreground contention on the single main slot — `targets_main_node`
computed from pool *presence* not routing *outcome*, and consumers omitting `off_main_only=True`, so a
worker hiccup dogpiles the foreground slot. **This session FIXED the delegation/swarm cluster** (most
corroborated + live-log confirmed + self-contained); the contention theme, the streaming fail-open,
and the router items are logged for follow-up.

Fixes shipped (each verified against source before editing; tests
`test_delegate_swarm_review_fixes.py`, 6 green):
1. **CRIT — `delegate(wait=True)` deadlock** (corroborated by 2 agents, independent traces). A
   sub-agent's LLM calls are forced `is_background=True` and park on `_wait_for_foreground_clear`
   while a user request is active; `wait=True` blocks the parent inside `reg.wait` while it still
   holds `foreground_requests` up → circular wait → ~600s dead air then a timed-out FAILED job (the
   same self-stall shape the compaction/context-shield fixes named, via a new entry point). Fix:
   `tool_delegate` downgrades `wait=True`→fire-and-forget when `foreground_requests > 0` (true iff an
   interactive turn is in flight; false in idle/autoadvance where wait is safe), returns the job ids +
   an explanatory note, model collects next turn. `tools/delegate.py`.
2. **HIGH — `delegate_to_swarm` advertised while unconfigured** (matches the live strike-burn). Gated
   out of `get_active_tool_definitions` when `llm_client.swarm_clients` is empty (mirrors
   `image_generation`); dispatch entry kept so a hallucinated call still gets the "process
   synchronously" steer. `tools/registry.py`.
3. **HIGH — stale `"delegate" → "delegate_to_swarm"` alias** hijacked case/paren variants of the real
   `delegate` tool (alias table consulted before the exact `norm_to_real` match). Removed the alias;
   the swarm-specific aliases stay. `core/agent.py`.

Docs: `docs/tools/swarm.html` (schema gating), `docs/core/delegation.html` (wait=True downgrade).
NOT yet deployed as of this entry — plain-kill to deploy.

### 2026-07-20 (later 3) — three-stack review FIXES: 2 crit + ~12 high + ~20 med, all fixed in the recommended order

Ten parallel fix agents (strict one-file-per-owner) + coordinator on `core/agent.py`, `main.py`, and
`interface/slack_project_commands.py`. Every §4B "2026-07-20 three-stack review" finding addressed;
each fix batch shipped its own regression tests. Full suite green. By the recommended fix order:

1. **H1 streamed-finalize bypass (headline + self-correction).** Streamed forced-final turns `return`
   the SSE generator before `_finalize_and_return`, so `add_work_log` + `_record_turn_trajectory`
   (both sole-called from finalize) never ran — and yesterday's Round-10 streamed-trajectory backfill
   was INERT (keyed off a trajectory finalize never wrote). Fix: extracted the work_log write into a
   shared `_write_project_work_log_safe` helper called from BOTH finalize and the stream drain, and
   the drain now calls `_record_turn_trajectory` directly with the real `full_content` (records +
   outcome-heuristics + correction-stash in one; the late verifier backfills the outcome by
   trajectory id). Removed the dead `_streamed_traj_pending` machinery. Tests:
   test_project_work_log.py (helper pins + both-paths-call-it), test_turn_loop_review_fixes_20260720.py.
2. **C1 cleanup sandbox escape + H8/H9 cleanup gaps** (workspace_cleanup.py + tools/projects.py +
   memory/projects.py): `_escapes_projects_root` containment assertion in sweep/tidy, `cleanup` routed
   through `_resolve_project_ref` at the tool boundary, `.git`/dot-config never debris + age-gate on
   all tidy debris, and `_normalize_rel`/`register_file_artifact` aligned so absolute
   `/workspace/projects/<id>/x` deliverables are protected.
3. **C2/H6/H7 write-corruption trio**: auto-promote + fence strips pass `filename=` (sanitizer guard
   armed); native leaked-framing repair won't truncate a body-sized value on sibling-only evidence
   (agent.py:1553); streaming-replace (>1 MB) routes through the marker-leak + syntax-regression guard.
4. **H5 verify fail-open + H10 no-runner + H3 reaper**: advancer got a fail-closed verify classifier
   (grep-no-match / egress-prose / no-exit-code → inconclusive, never DONE) and refuses to DONE a
   runner-less tool leaf; the API route and Slack `advance_async` now build a real project-pinned
   runner (or refuse) instead of the classify-only path; `ProjectStore.reset_orphaned_in_progress`
   wired at boot in main.py resets stale IN_PROGRESS claims left by a deploy/crash.
5. **H2 metadata whole-dict replace**: `update_project` metadata is now MERGE-by-default inside one
   BEGIN IMMEDIATE txn (explicit `metadata_replace=True` for a true replace) — the documented
   budget-raise no longer wipes ledger/config/counters.
6. **H4 prune third return + the medium tier**: the `<=5`-message branch now caps via
   `_cap_oversized_tail`; pruning reserves per-turn injection headroom (`_history_budget =
   max_context - min(24k, max(0, max_context-32k))` — the reserve ramps in only above a 32k window so
   it never lowers a small/mock test budget's threshold, which had perturbed compaction timing and
   flipped the `execution_failure_count==4` System-3 pivot under the loaded-tokenizer full-suite
   ordering); scratchpad injection capped at source; batch dedup no
   longer collapses stateful tools (`_BATCH_COLLAPSE_UNSAFE`); `is_mutating` aligned with
   `is_sandbox_mutation` (unzip/git_clone/image_generation); post-batch failure bookkeeping attributes
   to the FAILING tool not the batch's last; ReplanBridge now PERSISTS via `ProjectPlan.request_revision`;
   plan-gate coerces string/null fields; HUMAN_GATE → NEEDS_USER; contradiction/digest/concepts,
   execute exit-codes + heredoc guard, coding_executor append/retry/insert, and the API/route
   canonicalization + off-loop blocking — all fixed.

Deferred to docs owner: HTML docs for the new contracts (metadata merge, `metadata_replace`,
`human_approved`, `ProjectPlan.request_revision`, verify-classifier, advance 503, the
`projects/<other-id>/` explicit-foreign-path behavior change). **Docs DONE 2026-07-22** (3 parallel
doc agents; several sections turned out to already exist from the fix session — the genuinely missing
pieces were project_safety.html's human-gate/NEEDS_USER contract, the advancer boot-reaper bullet,
the Slack `advance_async` runner parity, the REST-visible metadata-merge bullet, and the tool-surface
notes in tools/projects.html; every pre-existing claim was re-verified against source. Notable
contract nuances now documented: `metadata_replace` is store-API-only — neither `manage_projects`
nor REST expose it, so merge is the only model/REST-reachable behavior; the boot reaper resets
orphans to READY while the in-tick no-runner release resets to PENDING; ReplanBridge lives in
core/triggers.py:327, metacog.py only wires it.) **DEPLOYED 2026-07-22** — suite green pre-deploy
(8586 passed / 12 skipped), plain-kill, respawned healthy in ~10s (health OK, all 3 nodes attached).

### 2026-07-20 (later 2) — three-stack review (project-autonomy + turn-loop + code-correction): LOGGED, not fixed

Ten parallel review agents over the next-stalest stacks (last dedicated sweep 07-03/04; heavy churn
since). Every finding re-verified against source by the coordinator, criticals reproduced; deduped
against §4B/§4C. Result: **2 critical, ~12 high, ~20 medium — all logged in §4B's 2026-07-20 cohort
block, NONE fixed yet** (user asked to log for later pickup). The `TurnState`/`FinalizeState` seam
audited CLEAR (no `locals()` recurrence).

**Headline + self-correction:** the biggest finding (H1, corroborated by 3 agents) is that streamed
forced-final turns `return` the SSE generator at `agent.py:12019` BEFORE `_finalize_and_return`
(13164), so `add_work_log` and `_record_turn_trajectory` (each sole-called from finalize) never run
on streamed turns — web UI always streams and one-task-per-turn guarantees the streamed ending on
task-closing turns. **This makes the Round-10 "streamed turns backfill final_response" fix (below,
claimed HIGH-fixed) INERT** — the parking setter is gated on a non-str `final_content` its only
caller never passes, backfilling a trajectory that streamed turns never create. Correct fix = move
the work_log + trajectory-record + correction-stash into the stream drain, plus a streamed-turn
integration test (its absence let the inert fix pass green). Other criticals: `manage_projects
cleanup project_id="../.."` sandbox escape into destructive delete (reproduced); replace→write
auto-promote overwriting a file with an inner code snippet (missing `filename=` guard). Full list +
fix-order in §4B.

### 2026-07-20 (later) — metacognitive-stack review: 2 criticals + ~15 majors found and fixed same-day

Eight-slice parallel review of the whole metacog stack (~20k lines: metacog signals, reasoning
search, dream/REM, failure learning, reflection+verifier, selfhood, distill/PRM, agent.py wiring),
every finding adversarially re-verified against source before acceptance; then ten parallel fix
batches with strict file ownership + a coordinator pass over agent.py/main.py. Full detail (with
severity table) in `docs/audit_fixes.html` Round 10; new tests throughout. Highlights:

- **CRITICAL — counterfactual replay was a total no-op since its 2026-07-17 ship**:
  `persist_challenge` required exact `"SUCCESS"/"FAILURE"` but dream passes decorated strings
  ("SUCCESS (in 2 attempts)") — zero challenges ever persisted (`$GHOST_HOME/system/counterfactual/`
  didn't exist). `classify()` had the same exact-match bug (a fixed persist alone would have graded
  every successful replay a REGRESSION and quarantined good lessons), and infra-failed replays
  read as decisive failures. All three fixed (prefix-normalize; "inconclusive" branch: no
  quarantine, bounded retry); quarantine attribution now prefers the sim-scoped
  `dreamer.last_selfplay_hydrated_triggers` snapshot over the shared attribute a concurrent user
  turn can re-stamp.
- **CRITICAL — redaction bypass into the training corpus**: lowercase/colon secret forms
  (`password: hunter2`, `db_password = hunter2`, `api_key: sk_live_…`) passed every redact.py rule
  → unredacted secrets in `$GHOST_HOME/trajectories/` and GHOST_LLM_RECORD fixtures. New
  case-insensitive assignment rule; escape-aware JSON rule reordered FIRST (integration caught the
  new rule truncating `"a\"b_secret"` at the escaped quote). Selfhood's `redact_pii` had the inverse
  bug (redacted "skateboarding", missed `api_key=…`) — both directions fixed.
  **Follow-up DONE (same day): `scripts/scrub_secrets.py` (dry-run/--apply, per-file backups,
  live-append-safe) scrubbed the corpus — 4 genuine leaked values in 2 trajectory files
  (a `"password":` JSON value + a `SECRET = "…"` snippet); verified clean after. The dry-run also
  exposed a CC-rule false positive — 13+ digit runs INSIDE 32-hex trajectory/experience ids
  Luhn-passed and would have been corrupted — fixed in BOTH redactors with letter-excluding
  boundaries (`(?<![0-9A-Za-z])`) + tests. Scrub backups (containing the originals) left at
  `$GHOST_HOME/system/pre_scrub_backup_1784546621` — operator: delete after review.**
- **Finalize-extraction regressions**: the plan-postcondition gate and the selfhood
  reference-counter both read `locals().get(…)` from inside `_finalize_and_return` — dead since the
  step-3 extraction, no log. Now FinalizeState fields (`current_plan_json`, `wakeup_prefix`),
  test-pinned. Same idiom audited across agent.py — only these two sites were broken.
- **Verifier fail-open**: non-dict judge reply → AttributeError → whole pass silently skipped;
  truncated stage-2 fragment → fabricated UNCERTAIN@0.5 that SUPPRESSED the classic fallback;
  `GHOST_VERIFY_STAGE_NO_THINK=0` defeated by an ungated `stop=["\n"]`. All fixed
  (`_parse_json` guarantees dict; verdict key required; stop inside the guard).
- **Self-play mutated production skill state**: dream's ReadOnlySkillMemory `__getattr__`
  passthrough let the sim's bus bump real retrieval counters (prune-eligibility pollution) — now an
  explicit whitelist like the vector wrapper's M1 fix. Related: `helpful_retrievals` double-credit
  closed (judge stamps `last_credited_at`; both live `credit_recent_retrievals` calls now use the
  discriminative query/trigger form) — this also un-corrupts the §3 distilled-lesson kill-criterion
  audit.
- **Frontier ABBA lock inversion** (event-loop deadlock risk) fixed + AST lock-order test.
  **Async-critic timeout** now hands the running verdict to the late handler instead of abandoning
  it AND double-spawning; late-verdict side effects moved to spawn_bg. **Streamed turns**: calib
  stash tagged by req_id (cross-request mispair race); drain now backfills `final_response` into
  the trajectory (user-correction promotion + abort-marker detection were dead for ALL streamed
  turns). **Journal-mined self-play challenges** could never fire (phase-1 drains at ~2min idle,
  phase-3 mines at >60min): phase-1 now tees mineable post-mortems into a bounded stash
  (`system/selfplay/journal_stash.json`) that generation falls back to; dream's raw journal drain
  removed. **PRM default checkpoint** now loads at boot (was orphaned on every restart).
  **Uncertainty tracker** made real: footer before blanket-verify, resolutions persisted,
  request-start reset. Plus ~35 minors (see docs Round 10 table).
- **Test-isolation incident (fix immediately deployed)**: the dev shell exports GHOST_HOME, so
  today's dream/self-play test runs wrote 112 synthetic challenges into the LIVE counterfactual
  ledger and unrelated tests replayed from it. Ledger purged (backup in session scratchpad;
  everything in it was test-minted — the live agent runs pre-fix code that cannot persist), and
  tests/conftest.py gained an autouse `delenv("GHOST_HOME")` fixture so no test can resolve live
  operator state again.
- **NOT yet deployed**: the live agent still runs pre-fix code — plain-kill the launchd service to
  deploy when ready. On first deploy expect: counterfactual ledger starts populating (real entries
  this time), PRM loads any idle-trained checkpoint at boot, boot log may now print
  `arbiter=off (module toggle)` (accurate — the bundle flag was misreporting).

### 2026-07-20 — overnight log-eval fixes: git-conflict-dialect mis-split + verifier mutated-file coverage + dream churn cap
Night-log review (boot 07-19 20:54 → 07-20 06:50) found one serious incident and one
systemic waste loop; both root-caused from the trajectory and fixed, full suite green
(8236 passed). WebOS games click-verified end-to-end (the request the loop-breaker cut
short at 22:29) — two REAL game bugs found and fixed in the process.
- **Marker-leak RECURRENCE root-caused** (request 97e42cea, 21:59): the model emitted a
  *well-formed* envelope in the 7-char git-conflict dialect (`<<<<<<< SEARCH` /
  `=======` / `>>>>>>> REPLACE`). The strict parser only knew exact-4, so the payload
  fell into the loose fallback whose separator regex matched `====` ANYWHERE — the
  `=====` inside the banner comment on the SEARCH text's first line became the
  separator, SEARCH parsed as `//`, and the file's first `//` (inside the CSS
  `@import url('https://…')`) was replaced with the whole marker-stripped blob,
  reported SUCCESS. All 4 of the 07-14 guards were structurally blind to it (CSS
  damage ≠ JS parse failure; exact-4 `_MARKER_LINE_RE` didn't know `=======`).
  **Fixes (tools/file_system.py):** width-tolerant (`<{4,}`/`={4,}`/`>{4,}`, optional
  `REPLACE`) + line-anchored-separator parsing in BOTH regexes; width-tolerant
  `_MARKER_LINE_RE` (reverses the RST-underline carve-out — count-awareness covers
  it); mis-split sanity (SEARCH < 3 chars or containing marker lines → reject);
  regex-based envelope detection (no-space variants). Deliberate behavior change: a
  separator glued to content (`…')====`) now fails CLOSED instead of parsing.
  Tests: test_replace_marker_leak_guard.py (24), test_file_system_replace.py updated.
- **Verifier mutated-file coverage (core/agent.py):** the corruption shipped behind
  CONFIRMED (100%) because WEB-EXEC probed only the FIRST located entry page
  (minesweeper.html, not the mutated index.html) and FILE-ARTIFACT only re-read
  prose-claimed ("created …") files. Now `_execute_web_artifact` probes EVERY html
  written/replaced this turn (cap 4; any located page failing to navigate →
  inconclusive, never "clean") and FILE-ARTIFACT checks claimed ∪ mutated (new
  `_files_mutated_this_turn` collector). Honest limit: CSS-only damage still loads
  clean — the parser guards above are the primary defense. Tests:
  test_verifier_web_exec.py, test_grounded_file_verify.py, test_verifier_auto_repair.py.
- **Dream-churn cap:** overnight REM re-extracted the same two heuristics 10+ times
  (0 meta-memories all night) — each idle self-play run minted ONE new digest ID,
  reopening the equality-only idempotency guard over a 59/60-identical window, and
  reworded re-saves (measured 0.07–0.17 apart on the live bge-small store) slipped
  past the 0.15 dedup cutoff. **Fixes:** delta-aware guard (`REDREAM_MIN_NEW_FRAGMENTS`
  = 3, env `GHOST_DREAM_MIN_NEW`; un-dreamed fragments accumulate — cache key updates
  only on success) in core/dream.py; mistake-less rules dedup at < 0.25 (distinct
  rules measured ≥ 0.29; env `GHOST_RULE_DEDUP_DIST`) in memory/skills.py. Tests:
  test_dream_selfplay_seeds.py, test_dream_async.py, test_skill_dedup.py.
- **WebOS games actually fixed + click-verified** (Playwright, 15/15): the overnight
  "remove duplicate gameLoop() calls" edit removed ALL gameLoop call sites — the
  platformer froze at menu (Enter changed state, nothing animated). Re-added the start
  in the menu→playing branch with a cancelAnimationFrame guard (loop-continuation
  condition covers the other states). Minesweeper's `getLevelConfig()` read the
  globals `rows`/`cols` before init assigned them → NaN mines → ZERO mines placed →
  first click flood-won instantly (self-heals on game 2, hence missed). Fixed to
  `base.rows`/`base.cols`. Verified: icons open windows, grid reveals correctly,
  Enter starts the game, levelComplete overlay draws, Enter advances level 0→1,
  single ~60fps loop.
- Docs: tools/file_system.html, core/verifier.html, core/dream.html, memory/skills.html.

### 2026-07-19 — harness-dimension failure attribution + failure-cluster distillation
MemoHarness adaptation (arXiv:2607.14159): every failure record now names the harness
layer it was attributed to, and recurring failure clusters distill into preventive
pattern lessons during REM. Motivation: three multi-day debugging episodes (verifier
evidence-packer truncation, fail-open SEARCH/REPLACE parser, native-tools arg
corruption) were all harness failures initially blamed on the model.
- **`core/failure_dimension.py`** (new, leaf): 8-value taxonomy
  (context_assembly / tool_interaction / generation_control / orchestration / memory /
  output_processing / model / unknown), first-match-wins regex tables ordered
  harness-before-model ("audit the harness first"; network ETIMEDOUT →
  tool_interaction, LLM ReadTimeout → generation_control), worker adjudication via
  new `CLASSIFY_FAILURE` route label. Env helpers read per-call.
- **Lessons**: `dimension` field on the schema (additive, `_normalize_lesson`
  back-fill, vector-twin meta); auto-classified at the `learn_lesson` chokepoint for
  real-mistake lessons (mistake-less rules stay empty; explicit kwarg wins; dedup
  merges back-fill it). No producer call-sites touched.
- **Work logs**: dispatch loop captures failure heads next to
  `classify_tool_failure` (`context._turn_failure_texts`, ≤6/turn); finalize
  classifies them (+ verifier reason) into the payload's `failure_dimension`
  (only non-completed outcomes pay).
- **`core/failure_distill.py`** (new): REM pre-gate pass — corpus from playbook +
  all-project failure work_logs + counterfactual regressions (deterministically
  `memory`), grouped by (dimension, `frontier.classify_cluster`), ≥3 cases →
  ONE pattern lesson via worker route `DISTILL_PATTERN` (400 tok, 60s), written
  through `learn_lesson` with `source="distilled"` so EXISTING hydration retrieves
  it — zero new read-side plumbing. Verbatim trigger reuse
  (`distilled(<dim>/<cluster>):`) → freq bump not row growth; evidence-fingerprint
  watermark in `$GHOST_HOME/system/failure_distill_state.json` skips unchanged
  clusters; unknowns adjudicated (≤8/cycle) and persisted. Cap 2/cycle.
- **`project_dream_pass` finally wired** (built-but-unwired since Phase 5): called
  from `dream()` alongside the distill pass, both BEFORE the entropy/idempotency
  gates (episodes-pass rationale). Gained the event-id watermark wiring it demanded
  (re-digestion spam otherwise) + failure work_logs join the digest with a
  `failures` count. LAST DREAM DIGEST briefing line now actually populates.
- Gotcha for future readers: the dimension tag is a **prior, not a verdict** — regex
  sees where a failure *surfaced*, not where it originated (the evidence-packer bug
  would have tagged `orchestration` from the refute text). The aggregate
  distribution is the reliable signal; §3 lists the kill criteria.
Toggles (all default ON, per-call env read): `GHOST_FAILURE_DIM`,
`GHOST_FAILURE_DISTILL`, `GHOST_FAILURE_ADJUDICATE`, `GHOST_FAILURE_DISTILL_MAX=2`.
Tests: `tests/test_failure_dimension.py` (23), `tests/test_lesson_dimension.py` (10),
`tests/test_failure_distill.py` (12), `tests/test_dream_failure_distill_wiring.py` (4),
+ additions in `test_project_work_log.py` / `test_project_advancer.py`. Docs:
`docs/core/failure_dimension.html` (new), `docs/core/dream.html`,
`docs/algorithms/dream_cycle.html`, `docs/memory/skills.html`,
`docs/core/project_advancer.html`, `docs/index.html`.

### 2026-07-19 — log-eval fixes: REM input starvation closed + reference gate fail-closed
22.5h log eval (clean day: 30/30 requests finished, 20/20 self-play solved, zero
tracebacks) surfaced two structural gaps; both fixed:
- **REM idle-spinning** (item 1): 40 REM cycles entered, 38 skipped "fragment set
  unchanged", `Auto-memory pool thin (0)` every single cycle. Root cause chain: auto
  pool has ~zero inflow (smart-memory ≥0.9 gate — known, journal §6 2026-07-09) AND
  the trajectory-digest fallback only refreshes on REAL requests (self-play detaches
  the collector by design), so overnight hourly self-play never changed the digest
  window. Fix: `selfplay_dream_fragments()` (core/dream.py) digests the frontier
  tracker's per-cluster `recent_outcomes` (cluster/passed/attempts/mistake, ids
  `selfplay:<cluster>:<ts>`, newest 20) and `dream()` merges them into the thin-pool
  fallback — a new self-play outcome reopens the idempotency guard, no new material
  still skips. Watchdog eligibility gate (agent.py phase 2) mirrors the new fallback.
  Self-play ids never reach the merge/delete pass (same contract as `traj:`).
- **Reference gate fail-open** (item 2): 3 challenges accepted that day with
  "Model omitted <reference_solution> — consistency gate SKIPPED", and the night's
  only solver failure was a validator ordering quirk (first-encounter order expected,
  never stated — solver's sorted output arithmetically correct, judged FAIL; the
  minted "insertion order" lesson at conf=1.00 generalizes that quirk). Fix, in the
  gen loop (core/dream.py): omission on a DATA-BACKED challenge (setup writes literal
  files) now triggers a targeted ~10s repair (regenerate ONLY the block,
  stop=</reference_solution>, mirrors validator repair); unusable repair → attempt
  REJECTED into the regeneration loop (fail-closed). No-data challenges stay exempt
  (nothing for the validator to disagree with). Plus generation CRITICAL REQUIREMENT
  #14 EXPLICIT OUTPUT ORDER: multi-line/multi-pair output must state its exact
  ordering in the challenge prompt — the quirk-lesson source class, cut off at
  generation.
- Tests: tests/test_dream_selfplay_seeds.py (new, 9) + test_selfplay_reference_gate.py
  (+5: fail-closed pins, repair/reject/exempt functional paths). Docs:
  docs/core/dream.html (§4b fail-closed, Self-play-seeded REM). Needs restart to go
  live (dream.py/agent.py are hot only on boot).

### 2026-07-18 (later 9) — Rick Dangerous churn loop: wrong refute + reopen/advance grind (FIXED)
Operator: "fix the wrongly refuted / auto advance loop that grinds turns." Diagnosis from
event log + trajectories: (1) user asked "restart the service" (req 4836cc14); verifier
REFUTED the correct reply with "the project is already complete — 14/14 tasks done" —
TASK-LEDGER STATE USED AS A VERDICT (the manage_projects listing rode along in the
evidence window); (2) that refute queued a correction banner; each surfaced banner led
the next turn and drove more "corrective" work → refuted again; (3) independently, a
user bug report reopens the DONE project (defect task, DONE→ACTIVE) and
--autoadvance-idle immediately grinds the reopened project (project_reopened 19:22 →
autoadvance_failed 20:18 → rollup DONE 20:33 → next refute reopens...). Three brakes:
- **Verifier prompts** (classic + stage-2 adjudicate): bookkeeping/ledger state ("all
  tasks done", "project complete") is NEVER by itself grounds for REFUTED; operational
  asks (restart/check/fix/run) judged on their own evidence. VALIDATED live on nova:
  the exact failing shape (new seed case service-restart-done-project, #13) now
  CONFIRMED conf=1.00 on BOTH arms.
- **Correction dedup** (_record_late_verdict): identical (note, conversation) banner
  queues at most once — repeated identical refutes can't stack banners that feed the
  loop.
- **Defect-reopen cap** (_note_defect_on_done_project): max _DEFECT_REOPEN_CAP=2
  reopens per rolling 24h per project, enforced via one _atomic_metadata_update
  (cross-process safe); past the cap → loud WARNING + no reopen, operator decides.
- Ruled out: advancer re-grinding FAILED tasks (next_ready_leaf selects PENDING/READY
  only); interface/server retry loops (one upstream POST per client send; resume replays
  the buffer). NEEDS RESTART to go live; until then the queued wrong correction for the
  Rick conversation surfaces once more.
- Tests: work_log +2 (cap + window expiry), streaming gate +2 (dedup), two-stage +1
  (prompt pins); seed set 13 cases. Docs: core/verifier.html.

### 2026-07-18 (later 8) — verifier two-stage prompt + fault-injection calibration bench
From the "Mechanisms of Introspective Awareness" paper (arXiv:2603.21396): yes/no detection
probes are dominated by a default-No gate that suppresses latent signal; forced
identification extracts it, and detection is only meaningful measured against a controlled
false-positive rate. Two changes:
- **Two-stage verify_claim** (core/verifier.py): stage 1 FORCES the judge to name 1–3
  weakest fragments (quote + alignment/support/constraint/artifact, sanitized to 3×300
  chars); stage 2 adjudicates each suspect against the evidence under the strict rubric
  (all outputs together; paraphrase ≠ fabrication; truncated-but-consistent evidence →
  UNCERTAIN at most, never REFUTED — the packer lesson). Same verdict JSON out, so the
  ≥0.7 gates/repair/late-verdict paths are untouched; suspects ride on
  VerifyResult.suspects. Fail-safe fallback to the classic single prompt on any stage
  failure. Kill switch GHOST_VERIFY_TWO_STAGE=0 (read per call). Cost: one extra
  VERIFY-route call per claim verdict; code/visual paths unchanged.
- **Fault-injection bench** (eval/verify_bench.py + scripts/verify_bench.py +
  scripts/verify_bench_cases.jsonl, 12 seed cases): inject KNOWN corruptions (fact_swap,
  fabrication, wrong_topic, silent_failure, artifact_leak, constraint_violation → expect
  REFUTED; evidence_truncation → expect NOT refuted) into clean triples, run against the
  live judge endpoint, report per-fault TPR + clean-case FPR, raw AND at the production
  ≥0.7 actionable gate. Mints extra cases from GHOST_LLM_RECORD day-files by inverting the
  rendered claim prompt (template-derived literals, survives template edits).
  `--two-stage both` A/Bs the new prompt vs. classic per fault class.
- **Live-judge probes forced a stage-call payload discipline** (measured on Nova/Gemma 4
  E4B heretic): raw stage calls cost 600–1200 tokens / 30–90s per 60-token verdict via
  three failure modes — nondeterministic `<|channel>thought` prelude (adjudicate prompt),
  fenced/pretty JSON, and verdict-looping to the cap (server runs repeat_penalty=1.0).
  Fix: no-think soft+hard switch (GHOST_VERIFY_STAGE_NO_THINK=0 to disable), stop=["\n"],
  "MUST start with {" prompt line, GHOST_VERIFY_STAGE_MAX_TOKENS=1024 cap. After: enum
  ~41 tok/3–5s, adjudicate ~63 tok/~4s, 100% valid JSON. response_format=json_object was
  tried and REJECTED — grammar sampling made the judge MORE verbose (41→786+ tok,
  cap-truncation). NOTE: the CLASSIC deployed prompt measured 637 tok/42s on the same
  judge — brushing the 45s worker timeout; two-stage with discipline is now FASTER than
  classic (~8–12s vs ~42s). The 2026-07-16 "7–11s uncontended" verdict figure no longer
  reproduces on nova.
- **First live A/B (2 seed cases vs nova) — the bench caught its own feature regressing.**
  Classic arm: TPR 0.6–0.75 / FPR 0 at ~40s/verdict. Disciplined two-stage v1: TPR 0.42
  (terse stage 1 named one trivial suspect, missed the injected fabrication; stage 2 only
  judged named suspects). v2 fixes: stage 1 = EXACTLY 3 suspects incl. ≥1 word-by-word
  checked fact; stage 2 = per-suspect "checks" array generated BEFORE the verdict keys
  (structured deliberation replacing the removed free-think) + may flag beyond-suspects
  problems. Re-measured: **TPR 0.75 / FPR 0.0 at ~16s/verdict — classic catch rate at
  ~2.5× speed**. Default stays ON.
- **Judge-level blind spots (every arm, prompt-independent), the real backlog:** (a)
  injected diff-markers (artifact_leak) 0/6 caught — CONFIRMED conf 1.0; consider a cheap
  regex artifact gate before the LLM verdict; (b) single-digit fact swaps (34→35°C)
  usually missed; (c) truncated evidence draws spurious REFUTEs in all configs (the
  packer shape, now QUANTIFIED — rubric wording doesn't fix it); (d) confidence pinned at
  0.9–1.0 everywhere → the ≥0.7 actionable gate is effectively verdict-only on this judge.
  Full 12-case A/B (~90 min on nova) pending — operator to schedule.
- **POST-DEPLOY DISCOVERY — the verifier gate never runs on STREAMED turns.** Operator
  asked why a project turn got no verdict: log had no verdict/late/skip lines (the late
  recorder logs every outcome), USR2 task dump showed no verdict task, and the code shows
  `_compute_verifier_verdict_gated` is called ONLY from `_finalize_and_return`
  (non-streaming) + the in-loop repair block on that same path — the streaming closure
  (decomposition step-4 leftover) has zero verifier calls. Interface turns stream → every
  streamed answer ships unverified, silently. Yesterday's 9C late-refute came via a
  non-streaming client. NOT caused by the two-stage change or the restart — pre-existing
  coverage gap, unwired-loop shape.
- **FIXED same session — streaming verifier gate wired:** stream_wrapper's post-drain
  bookkeeping now spawns `_compute_verifier_verdict` + attaches the late handler. Always
  late regardless of GHOST_CRITIC_ASYNC (text already shipped; zero client latency);
  `force_correction=True` threaded through `_attach_late_verdict_handler` →
  `_record_late_verdict` so a high-conf late REFUTED queues the next-turn banner even
  outside async mode (streams never get an inline note; banner already surfaces via
  `_take_active_correction`). Gotchas handled: outer finally deletes messages/tools →
  full shallow copy + EAGER `_conversation_fingerprint(messages)` at closure creation
  (messages[-10:] loses the first user msg the fp is keyed on → corrections would
  silently drop); toolless turns skipped (noise); <think> stripped from claim (2000-char
  budget). Tests: test_streaming_verifier_gate.py (+8 incl. source-wiring pins). Docs:
  core/verifier.html new section.
- **Live validation of the 19:21 deploy (2 findings):** (a) STREAMED requests split into
  TWO sub-paths — spontaneous-final turns (model just answers, e.g. the weather test req
  1b05b803) finalize via `_finalize_and_return` and re-stream, so they ALWAYS had the
  gate: 1B produced a real `LATE REFUTED (90%)` (claim cited Elefsis-station data for an
  Athens ask — two-stage on nova, e2e proof incl. lesson scrub + correction queue);
  forced-final turns (xrick auto-advance) take the live `stream_wrapper` path = the new
  gate. (b) The new gate stayed SILENT on xrick turns and every skip/except channel was
  logger.debug = captured nowhere → indistinguishable from never-ran. Gate rewritten
  LOUD: each skip branch (no claim after think-strip / no substantive tool / no verifier)
  + spawn-success + spawn-failure logs at INFO/WARNING, log-only. Activates on next
  restart (operator restarts at will; NO restart from assistant per operator
  instruction). Next xrick turn after that restart names the silent branch. Suite: 8143
  green.

### 2026-07-18 (later 7) — 9C postmortem: scope-flap heal + pivot timeout
Request 9c9b75aa (76 min, strike-cap death) ran on a boot WITH the later-6 trio: the
futility breaker fired correctly at +684s; the fatal was infra, not iteration:
- **Scope-flap**: `python3 extract_data.py` failed 4x with "can't open
  /workspace/extract_data.py" while the file sat in projects/7c990a4baf59/ — those calls
  ran WITHOUT the project workdir (stateful opts out of scoping by design; transient scope
  clears do the same). A `pwd` on a scoped call showed the right cwd → model diagnosed
  correctly 3x, retried the correct command, failed again. The remap heal was gated on the
  workdir kwarg = disabled exactly then. FIX (tools/execute.py): fnf + no workdir + canonical
  project id on the workspace-model mirror → one retry from /workspace/projects/<id> + note
  naming the mechanism (drop stateful or use absolute path).
- **System-3 pivot ReadTimeout burned 20 min** (+2886→+4101 dead air): generator+evaluator
  calls now timeout=120.0, fail-open.
- Working as designed in the same request: futility breaker (+684s), n-gram kill at 30k
  chars, "Same failure ×3" freeze, chunked reads, CWD-pin/steer (commands were bare
  relative — correct!), late verifier refute + queued correction.
- Tests: test_execute_path_and_exit1_heal.py +5 (§5 scope-flap), test_futility_breaker_trio.py
  +1 (pivot pin). Docs: tools/execute.html, core/agent.html.

### 2026-07-18 (later 6) — coding-struggle mitigations: futility breaker + syntax recipes + turn-budget honesty
Postmortem of requests 5b9fcc8f/39b4b62f/800a982d (xrick data extraction, 33 min + still
grinding): rewrote extract_data.py 5x/reran 5x with exit 0 every time (goal counts never met
— invisible to every failure-keyed breaker); failed 4 consecutive repairs of the same
`if c == '\':` unterminated-string bug and re-wrote the identical bug in the NEXT request;
n-gram thinking kill landed ON turn 40 so the grounding retry had no next iteration and
working narration shipped as the "answer" (late-refuted).
- **Futility breaker** (dispatch): per-request {writes, runs} per code-file basename
  (.py/.sh/.js/.mjs/.ts; runs matched via command text). 3 writes + 2 runs → ONE steer:
  record confirmed facts in ledger/notes NOW, verify ONE smallest unit end-to-end, switch
  approach CLASS. All-success goal-thrash finally has a detector.
- **Syntax-gate recipes** (file_system `_SYNTAX_FIX_RECIPES`): error-keyed RECIPE line
  appended to the write-time syntax warning — backslash traps ('\\' / chr(92)), line
  continuation strays, f-string brace nesting.
- **Turn-budget honesty** (handle_chat): `for/else` on the turn loop — natural exhaustion
  logs "all N turns used" and prefixes the reply with [TURN BUDGET EXHAUSTED] + "working
  state, NOT a finished result; ask me to continue from recorded findings".
- NOTE: the n-gram first-kill grounding retry already existed and worked — the 5B death was
  the turn cap, not a missing retry. Verified via turn count (40/40).
- Structural gotcha fixed during work: inserting a 20-indent block between two 24-indent
  siblings silently swallowed the off-project steer into the new except-handler suite
  (valid Python!) — caught by nesting assert, restructured.
- Tests: tests/test_futility_breaker_trio.py (new, 7). Docs: core/agent.html,
  tools/file_system.html.

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
