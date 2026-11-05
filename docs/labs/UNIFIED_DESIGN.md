# Unified design: workspace, labs, running things, projects (§4SX-b)

Status: **APPROVED 2026-11-03; reviewed r1 (author) + r2 (independent), all findings folded in (§12, §13)**. Names: the run lifetimes are `job | server | app` everywhere (older text saying task/preview is superseded).

Original status line: **APPROVED 2026-11-03** (operator answered D1–D8: all YES as recommended; D3 = narrow path; D6 "minimizing tool definition is good"). It builds on `docs/labs/PLAN.md` (labs on `disorder`).
Phase A of PLAN.md (the host service) is unchanged by this design except for the additions in §6 and can start
on 2026-11-04 as planned. Phases B–D change shape, see §8.

Evidence: three read-only inventories of projects, services and the sandbox, done 2026-11-03 (file:line in the
inventories; the key facts are quoted here), plus the live service registry and the trajectory counts.

---

## 1. What is wrong today (facts)

| # | Problem | Evidence |
|---|---|---|
| P1 | **Three ways to run something in the background**, all the same `setsid nohup` in the same container, with **two registries + an in-memory mirror**, each with its own liveness and reaping rules: `execute … &`, promoted **jobs** (`sandbox/jobs.py` says it "mirrors sandbox.services"), and **services** (`sandbox/services.py`, 2 869 lines). | services inventory §1, §Overlaps |
| P2 | **"Supervisor" never supervises.** A crashed service stays dead. A sandbox recreate kills every service, and the rows then read DEAD/STALE until the model notices. Health is checked once, at start (~6 s). | `services.py:1830-1870`, `:2641` |
| P3 | **Reaching an app is awkward.** Ports 8100–8104 are published on loopback only. Reaching an app from the tailnet needs the operator to run `bin/serve-remote.sh`. That makes 5 ports for all projects and hard-coded port rewrites. The §4EJ incident: the model used 8899, the report said "listening ✓", and it was never reachable. | `docker.py:960-990`, journal §4EJ |
| P4 | **High failure rate and regression churn.** `manage_services` has 86 owner calls, 22 errors (18 of them starts). §4LR: 20% failures, 6 MAJOR + 2 MAJOR regressions inside the fixes. §4SK: a stale row cost 4 turns, and 3 review rounds found MAJORs in "replace". | services §4–5 |
| P5 | **The registry is stale and messy.** 8 rows: port 8101 appears in 3 rows, `jiu-jitsu-calendar` is both scoped and unscoped, `minecraft-test` is on 8110 (unreachable), and 6 of 8 belong to no project. | live `registry.json` |
| P6 | **Registries live on the bind mount** (`/workspace/.services`, `.jobs`), so the sandbox can rewrite them. §4GI had to add symlink guards (a log symlink leaked the API key). | sandbox §6 §4GI |
| P7 | **A project only knows about its sandbox services.** Jobs carry an id parsed from the workdir path, scheduled tasks have no link, and labs don't exist yet. There is no single "what does this project own" record. Only services are cleaned up on archive/delete. | projects §5, §Summary 3 |
| P8 | **Autopilot can't operate running things.** `LEAF_ALLOWED_TOOLS = ("file_system","execute")`, so it has no services, no jobs and no labs. The briefing's SERVICES block is registry facts, never probed. | projects §4, §5 |
| P9 | **The sandbox does two jobs: workbench and production host.** Owner apps (chess-coach, jiu-jitsu-calendar, webos…) run in the same container as the agent's scratch work, share its 4 GB, and die when it is recreated. | sandbox §Summary |
| P10 | **One global project binding** (a scratchpad sentinel pair) with guests and holds. Ownership "theft" recurs. | projects §3, §7 |

## 2. The idea in one paragraph

Separate **where** things run, **what** is running, and **who owns it**.
There are exactly **two kinds of place**: the **workspace** (the agent's own desk: files, code, quick commands;
one, persistent) and **machines** on `disorder` (labs: containers the agent creates and destroys). Anything that
**keeps running** is a **run**, in one registry, wherever it runs, with one of three lifetimes. Every place and
every run has **one owner** (a project, or a conversation for ad-hoc work). A project's state is its **inventory**:
files, machines, runs, schedules. Every lifecycle action (archive, delete, release, autopilot, briefing) walks that
inventory the same way. Production apps move off the agent's desk onto kept machines.

## 3. The model (four nouns)

### 3.1 Places
| Place | What it is for | Lifetime | Network |
|---|---|---|---|
| `workspace` (today's sandbox, unchanged inside) | writing code and files, analysis, quick commands, **dev previews** | persistent | Tor (transparent, §4FU) |
| `lab_<id>/<member>` (a machine on `disorder`) | the system under test, clusters, anything that needs root/ports/network faults, **and deployed apps** | **disposable** (TTL) or **kept** (see 3.3) | Tor (transparent, PLAN §3b) or offline |

One rule for the model: **you build in the workspace and run systems on machines. Machines can vanish, so the
recipe and the results live in the workspace.**

### 3.2 Runs (replaces `execute &`, jobs, services)
A **run** is anything that keeps executing after the tool call returns. One registry, owned **host-side** by the
agent (sqlite under `<memory_dir>`, **not** on the bind mount, which fixes P6). Fields: `run_id, name, where, command,
lifetime, port, owner, origin, started_at, state, last_probe`.

| Lifetime | Today's equivalent | Ends | Port | Restart on crash |
|---|---|---|---|---|
| `job` (was `task`) | promoted job / `execute &` | when the command exits (or TTL 45 min, as now) | no | no |
| `server` (was `preview`) | sandbox service (dev) | when stopped, or the owner's project is idle/archived | leased | no |
| `app` | released app / "keep it running" | only when stopped or its project is deleted | leased, **tailnet URL** | **yes** (see D2) |

- `where` is **required** on every start. There is no default, so no silent misrouting.
- `execute` keeps auto-promoting a long command to a `task` run (behaviour kept; the registry changes).
- `execute … &` of a server stays blocked, now pointing at `run start`.
- In the workspace a run is the same `setsid nohup` mechanism as today (one launcher, not two). On a machine, a run is
  the labs service's detached job (`exec detach`). For `app` it is the **manager's** supervised process: the manager's health probe restarts it
  and counts crashloops (r2 #2: Docker's restart policy can't see an app crash, because the contract entrypoint ends in `sleep infinity`).
- **State is probed, not remembered:** `status` asks the place (pid alive + port answers; or the labs API), and the
  briefing shows the probed state (fixes P2's stale rows and P8's unprobed block).

### 3.3 Machines (labs), two lifetimes
- **disposable**: everything in PLAN.md (TTL 4 h/24 h ad-hoc; project labs 12 h/72 h pending confirmation).
- **kept**: no TTL. Created by `release` or an explicit "keep it". Has a restart policy, a fixed tailnet port and its own
  RAM reservation (D4). Destroyed only by the owner, by project delete, or by `unrelease`. Autopilot may **not** destroy
  a kept machine (a narrowing of D7 for production only; D5).

### 3.4 Owners and the inventory
- Every place, run and schedule row carries `owner = project:<id> | conversation:<id>` and `origin = owner | autopilot | probe`.
- **Inventory** = one query over files (the manifest), machines, runs and schedules for an owner.
  - `archive` stops `preview` runs, destroys **disposable** machines, keeps the recipes, and leaves `app` runs and kept machines up
    (they are production; ask in the summary whether to stop them).
  - `delete` removes everything in the inventory (the confirm preview lists it all).
  - `release` = build a kept machine from the recipe, start `app` runs, and **rehearse** (destroy and recreate from the
    recipe, then probe the URL through the host). That is stronger than today's restart-in-place.
  - Conversation-owned things are listed for that conversation and age out by TTL. "Destroy the site you just
    built" resolves inside the conversation's inventory.
- Scheduled tasks (`manage_tasks`, kept per operator) get an optional owner, so a project's schedule is in its inventory.

## 4. What the model sees (tool surface)

| Tool | Scope | Change |
|---|---|---|
| `execute` | run a command **now** in the workspace | unchanged (+ promotion goes to the new run registry) |
| `file_system` | workspace files | unchanged |
| **`lab`** | machines: `templates, create, list, status, exec (foreground), put, get, member <action>, logs, http, extend, destroy, spec, keep` | new (PLAN §5). `exec detach` removed → `run` |
| **`run`** (replaces `manage_services` + `jobs`) | everything that keeps running, anywhere: `start (name, command, where, lifetime, port?)`, `status [name│all]`, `logs`, `stop`, `restart`, `collect` (a task's output), `url` | new; one registry, `where` required |
| `manage_projects` | gains one view: `status` shows the **inventory** (files, machines, runs with probed state and URLs, schedules) | small change; no new actions |

Net change: two tools (`manage_services`, `jobs`) become one (`run`), plus `lab`. That's 29+ tools today → +1 net.
Aliases: `manage_services`/`jobs`/`services` → `run`, with a one-line hint (one release cycle only, then removed:
memory `workaround-outlives-its-bug`).

**Briefing (one block, KV-safe dynamic area, `build_project_briefing`):**
```
RUNNING (probed 12s ago)
  machines: lab_3f2a patroni-3 [pg1 pg2 pg3] disposable, expires 15:40
            lab_9c01 web (kept) → http://100.64.0.10:20003/
  runs:     api   app      lab_9c01/web1  :80   up 2d   http://100.64.0.10:20003/
            bench task     workspace            running 4m
```
For a conversation without a project, the same block covers the conversation's inventory.

## 5. Where apps live (the big move)

Today's owner apps run in the sandbox (P9). Proposal: **`app` runs live on kept machines on `disorder`**:
- Reached directly at `http://<labs host>:20xxx/` on the tailnet (D6; today's labs host 100.64.0.10). No `serve-remote.sh`, no 5-port cap, nothing
  on loopback (fixes P3).
- They survive agent restarts and sandbox recreates. Supervision comes from Docker's restart policy plus the manager's health
  check (fixes P2).
- Isolated from the agent's desk and from each other.
- The workspace keeps **`preview`** runs for quick dev looks (loopback, as today, with fewer ports needed).

**The one complication (D3):** apps that call the agent's API. Live: `projects/48e0373aaab3/app.py` (chess-coach) uses
`GHOST_SERVICE_TOKEN` against `host.docker.internal:8000`. Machines can't reach the tailnet (PLAN §3b, by design).
Options: (a) a per-machine, per-app **allowance to exactly `ghost:8000`** (one host-side forwarder on the lab network
→ the agent on `eva` (today 100.64.0.12:8000; **not** 100.64.0.13, which is the `ghost` image node, r2 #1), as an **HTTP proxy that allows only `/api/game/*`** (r2 #5), no other tailnet reach), or (b) such apps stay as `preview` in the workspace.
Recommendation: **(a)**, because it is narrow and explicit, the labs service records it, and it is tested like the egress rows.

**Migration of today's 8 registry rows** (P5): none has run since the 11-01 webos row. Proposal: chess-coach (project
48e0373aaab3) and jiu-jitsu-calendar (9009ff89d8bb) become `app` candidates. Ask the owner whether they want them
running. The other 6 are previews/tests and are dropped from the registry (files untouched).

## 6. Additions to the host service (PLAN Phase A)

1. Machine `lifetime: disposable|kept`. Kept machines have no TTL, `restart: unless-stopped`, a stable port lease (the same
   host port across recreate/rehearsal), and a RAM class.
2. RAM classes: of ≈4.9 GB, reserve **`kept_mb`** (proposed 1.5 GB) for kept machines; disposable labs use the rest.
   Owner pre-emption evicts only disposable autopilot labs, never kept ones.
3. Optional per-machine `agent_api: true` → the narrow forwarder of §5 option (a). Off by default, and a test proves the rest of the
   tailnet is still unreachable.
4. A health probe per `app` (HTTP or TCP, every 60 s). State is exposed in the lab detail. A crash-looping app is
   stopped after N restarts and reported (`crashloop`), not restarted forever.

## 7. What gets simpler (deleted)

- `sandbox/jobs.py` + `core/jobs.py` mirror + `sandbox/services.py` (≈2.9k + jobs) → one `runs` module with one
  launcher for the workspace and one adapter to the labs API.
- `.services/` and `.jobs/` registries on the bind mount, and their symlink guards.
- `serve-remote.sh` for apps (it stays only for previews, if wanted).
- The services' port-rewrite logic for the 5-port range (previews still lease, apps get tailnet ports from the labs service).
- `postgres_admin` (as already planned).

## 8. Phasing (replaces PLAN.md phases B–D)

- **A — host service on `disorder`** (PLAN Phase A + §6 here). *Starts 11-04 unchanged.*
- **B — `lab` tool** (PLAN §5) **plus the owner/inventory table** (machines first).
- **C — `run`**: the host-side run registry, the workspace launcher (moved from services/jobs), the lab adapter, `execute`
  promotion → `task` runs, aliases, briefing RUNNING block, removal of `manage_services`/`jobs`.
- **D — projects on the inventory**: archive/delete/release over the inventory, release → kept machine + rehearsal,
  the `lab` task kind, the autopilot leaf allowlist `("file_system","execute","lab","run")`, the `lab_capacity` hold, schedules with an owner.
- **E — cleanup**: remove `postgres_admin` and the Mac Postgres (PLAN §10.4), migrate/drop the 8 rows, docs.
- **B0 — project binding** (§10): per-conversation binding by `session_id`. Runs in parallel with A, and lands before B.

Each phase ships with tests, docs and an independent review. The suite runs once at the end. Deploy is gated on
`foreground_requests == 0`.

## 9. Decisions for the operator

| # | Question | Recommendation |
|---|---|---|
| D1 | Move long-running apps off the sandbox onto kept machines on `disorder`? | **Yes** (§5) |
| D2 | Auto-restart a crashed `app`? Your §4G requirement was "no auto-restart **on agent restart**". An app on `disorder` restarting on crash/reboot is a different thing, but it is your call. | **Yes for `app`** (with crashloop stop), **no for `preview`/`task`** (unchanged) |
| D3 | Apps that call the agent API (chess-coach): a narrow `ghost:8000` allowance, or keep them in the workspace? | **Narrow allowance** |
| D4 | RAM reserved for kept machines | **1.5 GB** of ≈4.9 GB |
| D5 | Autopilot may create/destroy disposable machines and runs freely (D7), but **not** destroy kept machines/`app` runs? | **Yes** |
| D6 | Replace `manage_services` + `jobs` with one `run` tool? | **Yes** |
| D7 | The 8 existing service rows: offer chess-coach + jiu-jitsu-calendar as apps, drop the other 6? | **Yes** |
| D8 | Project-lab TTL 12 h idle / 72 h max (still open from PLAN) | **Yes** |

---

## 10. Project binding (P10): redesign (added 2026-11-03 at the operator's request)

### 10.1 How it works today (facts, `tools/projects.py`)
- **Conversation identity = SHA-1 of the conversation's FIRST user message** (`conversation_fingerprint`, :815). The docstring
  says "the chat API … carries no conversation id". That is **no longer true**: web (`interface/static/app.js`, `sessions.js`),
  the CLI (`ghost` "Durable sessions: every turn carries a session_id"), the Slack bot (`slack:` ids, `api/routes.py:619-626`)
  and the ClockworkPi client all send `session_id`. The binding just never switched to it. Its own docstring (:840-847)
  admits a per-turn "Context:" preamble changes the first message and breaks the identity of the SAME session.
- **The binding is ONE global slot**: four scratchpad sentinels (`__current_project__`, `__current_project_conv__`,
  `__current_project_guests__`, `__current_project_held__`, :400-427) in a scratchpad with a **~50-entry / 24 h TTL LRU**,
  so the binding can be **evicted** (:1195-1203 handles "LRU-evicted" by failing closed).
- **`context.current_project_id` is process-global state** on the shared agent context, mutated at request start by
  `reconcile_conversation` (:1122) and read in **~110 places across 15 files**.
- Because there is one slot, a second conversation touching the project needs **guests** (§4QA), **held** conversations
  (operator 10-18: "a guest can't take the binding from the owner"), request-only switches with a long NOTE to the
  model, probe exemptions (§4PE), member save/restore (`core/agent.py:~30850`), a leaf-pin bypass (§4FH C1), a
  word-match "the message names the project" escape hatch (`_message_references_project`, a fuzzy rule), and park/unpark
  with scratchpad snapshot/hydrate. That is ≈600 lines (:400-1270), and the code carries **r9, r14, r15, r16, r17 MAJOR** fix markers. The
  journal mentions "binding" 224 times.
- What the owner experiences: "project belongs to another conversation", a project that drops out after the
  first message changes, ownership "theft" between owner/guest/probe, and switches that only last one request.

### 10.2 The root cause
Two things: **(1) the wrong identity** (a first-message hash instead of the session id), and **(2) a single global slot**
where the real relation is *per conversation*. Every guest/held/park rule is a patch for (2).

### 10.3 Design
1. **Identity = `session_id`.** The conversation key is `session:<id>` (Slack `slack:<…>` as it already is). The first-message
   fingerprint stays only as a fallback for a client that sends no session id (none of today's four).
   Probe/scheduled requests have no key (unchanged, memory `probes-have-no-conversation-key`).
2. **The binding is a durable per-conversation row**, not a global slot: table `conversation_focus(conversation_key PK,
   project_id, bound_at, last_used_at)` in the project store (sqlite, survives restarts, no LRU, no TTL eviction; rows idle
   > 30 days are pruned).
   - A conversation has **at most one focus project**. Many conversations may focus the **same** project (no owner, no
     guests, no held, no theft: a switch in conversation B only ever changes B's row).
   - `switch` / `create` / `resume` set this conversation's row. `exit` clears it. `archive` / `delete` clear every row that
     points at the project.
3. **Request scope is a value, not shared state.** At request start one `RequestScope(conversation_key, project_id,
   owner_key, origin)` is built from the row and carried in a `contextvars.ContextVar`. `context.current_project_id`
   becomes a **read-only property** over it (compatibility for the ~110 readers, migrated gradually). The leaf pin, the
   member turn (no project) and the probe (request-only) become three ways to **construct** a scope. Nothing is
   saved and restored around them, so the §4FH leaf bypass and the member save/restore disappear.
4. **No implicit activation by wording.** Delete `_message_references_project`. If a message names another project, the
   briefing adds one line: "this message names project X (id …); switch to it if you are working on it". The model
   switches with one call, and only this conversation changes. Tools that act on a project also accept an explicit
   `project_id` (reads, status, inventory), so looking at another project never needs a switch.
5. **The owner key of the unified design (§3.4) is this conversation key.** Ad-hoc machines and runs are owned by
   `conversation:<session_id>`. "Destroy the site you just built" resolves in this conversation's inventory, even
   across devices (a session resumed from the CLI is the same key).
6. **Scratchpad per project** (snapshot/hydrate on switch) stays, keyed by project as now. It is no longer tied to the
   global slot.
7. **Concurrency:** turns are serialized by the agent semaphore today. The ContextVar also makes background work
   (autopilot ticks, leaves) correct by construction if they ever overlap a turn.

### 10.4 What gets deleted
The four sentinels, guests/held lists, `_guest_of_bound`, `_guest_leaving_owner`, `_binding_is_anothers`, `_add_held`,
`_park_current_project` hygiene, the request-only NOTE, the fingerprint-mismatch escape hatch, the "no binding → fail
closed" branch, the member save/restore, and the leaf-pin bypass in `reconcile_conversation`. Roughly 500 of the ≈600 lines
go. The tests that pin guest/held behaviour are replaced by tests of the per-conversation row.

### 10.5 Migration
At deploy, the one live global binding can't be mapped from a first-message hash to a session id reliably.
**Drop it.** The owner's next message in that conversation re-switches with one call. Optional: map it by fingerprinting each stored
session's first message (the session store has the messages), and bind the one that matches. Cheap, but not required.

### 10.6 Decisions (binding)
| # | Question | Recommendation |
|---|---|---|
| B1 | Bind per conversation by `session_id` (many conversations may share a project; no owner/guest/held) | **Yes** |
| B2 | Drop implicit activation when a message names a project; replace it with a one-line hint + explicit switch | **Yes** |
| B3 | One focus project per conversation (switching changes focus; other projects readable by id) | **Yes** |
| B4 | Migration: drop the current global binding (optionally map it via stored sessions) | **Drop it, no mapping** (revised 11-03: the live slot holds ONE binding, project 41f745517fe4 "3-Layer Recursive Thinking…" + 1 guest; the cost of dropping is one re-switch) |

**Operator 11-03: B1 YES, B2 YES, B3 YES, B4 YES: drop the live binding, no mapping. Do it as part of Phase B0's deploy
(not before: the old code still reads the sentinels until B0 ships).** Step: after B0 is deployed, delete the four
`__current_project*` keys from `<memory_dir>/scratchpad.db` (table `scratchpad`), and confirm the next owner message
starts with no focus.

### 10.7 Where it goes in the phasing (§8)
**Phase B0 — binding**, agent-side only and independent of `disorder`. It must land **before** the inventory owners
(Phase B), because owners are conversation keys. It can run in parallel with Phase A (host). Order: A ∥ B0 → B → C → D → E.

---

## 11. The Docker host is a runtime setting (operator requirement, 2026-11-03)

`disorder` / `100.64.0.10` (labs host) and `eva` / `100.64.0.12` (the agent, the target of the D3 path) are **today's values**, never constants. (r2 #1: an earlier draft named 100.64.0.13, which is the `ghost` image node.) Rules:

**Agent side** (all read at startup, from args with env defaults, so the isolated-copy capability clear can null them):
| Setting | Env default | Meaning |
|---|---|---|
| `--labs-host` | `GHOST_LABS_HOST` | `host[:port]` of the labs service (tailnet IP or MagicDNS name). Unset → `lab` + `run where=lab…` are **hidden** |
| `--labs-key-file` | `GHOST_LABS_KEY_FILE` (default `~/Data/AI/.ghost_labs_key`) | API key, 0600 |
| `--labs-agent-api` | `GHOST_LABS_AGENT_API` | the URL **machines** use to reach this agent (D3 narrow path), e.g. `http://<ghost tailnet>:8000`. It is sent by the agent on each `agent_api` create, not baked into the host. Unset → `agent_api` apps are refused with a clear hint |

- **Visibility depends on configuration only, never on reachability.** Tools render before the system prompt, so showing or hiding
  a tool when the host goes up or down would re-prefill the cached head (§4LE/§4RC). Host down → the tool is still listed and
  returns `labs_unreachable` with a hint. Boot logs one line: labs host, version, health.
- **No address is ever composed by the agent.** Every URL comes from the labs API (`url` fields, built by the service from its
  own bind address). **Stored references are ids, not URLs:** the briefing, RELEASE.md, the inventory and notes store
  `lab_id` + container port and render the URL at read time from the current host, so a host change leaves no stale links.
- **A guard test** in the agent suite fails if `src/` contains a hard-coded labs host IP or hostname (the current
  values listed in the test as forbidden literals).

**Host side** (`ghost-labs` service): nothing host-specific in code or `install.sh`.
- `bind_host: auto` = `tailscale ip -4` (or an explicit value). The address pool, port range and RAM budget all come from config
  with auto defaults.
- **Images are built for the host's own architecture** (`uname -m`). Don't assume arm64 (disorder happens to be aarch64).
- `install.sh` is parameterless and idempotent on any Debian/Ubuntu host with Docker + Tailscale. It prints the API key
  and the `GHOST_LABS_HOST` value to set on the agent.

**Moving to another host:** install on the new host, then set `GHOST_LABS_HOST` + the key, then restart the agent. Disposable labs are simply lost
(they are disposable). Kept machines move with `ghost-labs export <lab>` (spec + named volumes as a tarball) → `import` on the
new host. That is Phase E, before the old host is retired. Until then, the agent's inventory marks kept machines whose host
changed as `missing (host changed)`. They are not silently dropped.

---

## 12. Review 2026-11-03 r1: findings and fixes

The independent reviewers were stopped by the account's usage limit, so this pass was done by the author against the code
and live data. **Run an independent review again (r2) before Phase B0 starts.**

### Code facts checked
- `jobs` / `core/jobs.py` now carry **only** `kind="sandbox"` (promoted execute). Delegates went with §4SI. Replacing `jobs` with
  `run` breaks nothing else ✅. But `is_promoted_result` (`sandbox/jobs.py`) is imported by `tools/acquired_skills.py:958`,
  `tools/composed_skills.py:199,346`, `tools/outcome.py`, `core/code_claims.py`, `core/build_gates.py`, `distill/outcome_heuristics.py`.
  It **must live on** in the runs module (same name, same result marker).
- `sandbox/services.py` is imported by **13 src files + `interface/voice.py`** (not only the tool):
  - `main.py` (boot reconcile, shutdown keep-container)
  - `tools/browser.py:646,917` + `tools/browser_runner.py:37` (loopback admission for service ports)
  - `tools/projects.py` (briefing, archive/delete stop, release rehearsal `:2620`, `create_version` `-v<N>` naming `:5441`)
  - `core/agent.py`, `core/autonomous_activity.py`, `sandbox/docker.py`, `sandbox/registry_guard.py`, `sandbox/jobs.py`
  - **`api/game_routes.py:74`: `valid_service_token` checks app tokens against the services registry.**

  Every one moves to the runs module in Phase C. The list is a floor (re-grep at C start).
- Binding facts (§10) re-verified:
  - `handle_chat` receives `body["session_id"]` (`core/agent.py:30544,30603`).
  - Web creates an id when absent (`app.js:2263`), per tab (`sessionStorage`) with a `localStorage` fallback.
  - CLI `chat_stream(…, session_id)`.
  - Slack sends `slack:<thread|dm>:<user>` (`slack_bot/main.py:1439`). The routes null it only for the **history store**
    (`routes.py:1333`), and the body keeps it, so it is a usable conversation key. The owner's Slack DM is one rolling
    conversation (one focus).
  - **Precedent:** the "state of work" (§4OE-1) is already keyed by `session_id`.

### Findings (each folded into the design; severity H/M/L)
1. **H, binding: the web console project routes use the last chat's key.** `api/projects_routes.py:117-141` `_set_current_or_held`
   switches with "the shared context still carries the last chat's conversation key", and `GET /api/projects` returns the
   global `current`. **Fix (B0):** the console routes take `session_id` (query/body) and act on that conversation's focus row.
   `current` is per session.
2. **L, dead code:** `interface/slack_project_commands.py` has no importers. Delete in B0 (it models a global `current_project_id`).
3. **H, phasing hole:** C removes services, but releases (rehearsal), the two live apps and the `valid_service_token` check
   still depend on them until D/E. If C ships alone, **chess-coach gets 403 and releases break.** **Fix:** C builds `run`
   *alongside* services. The tool switch, the token check move (now per `app` run, revoked by stop, the same semantics) and
   the services removal happen **at the end of D, after the two apps are migrated** (§8 updated below).
4. **H, release rehearsal would wipe app data.** "Destroy and recreate from the recipe" deletes named volumes. jiu-jitsu keeps
   its data in `data.db` next to the code, and chess-coach in `saves/` + `game_state.json`. **Fix:** a kept machine separates
   **code** (rebuilt from the recipe) from **data** (named volume `data`, mounted at `/data`, never removed by rehearsal).
   Rehearsal = recreate the containers with the data volumes kept, plus a fresh-from-recipe start of a **disposable twin** with an empty
   `/data` to prove the recipe alone works. Only `delete` (owner, project delete) removes data volumes, after an automatic
   final backup.
5. **H, how code gets onto a kept machine was unspecified.** **Fix:** `release`/`deploy` = (a) copy the project's deliverable files from
   the workspace to the machine's `/app` (`lab put` of a tarball), (b) install the recipe's deps **over Tor** (`requirements.txt` /
   apt list in the recipe, e.g. chess-coach needs `flask python-chess` + apt `stockfish`), (c) on the **first** deploy only,
   move data files (declared in the recipe: `data: [data.db, saves/, game_state.json]`) into `/data`. After that the
   machine's `/data` is the source of truth and the workspace copy is marked stale. **Updates** = redeploy code only
   (`/app` replaced, `/data` untouched). A **`python-app` image** (base + flask/gunicorn/common libs) speeds this up.
6. **M, stable URLs across versions and rollback.** **Fix:** the tailnet port is leased per **(project, app name)**, not per
   machine. `create_version` deploys to a new kept machine. The port moves to it only after its probe passes. The old
   machine is stopped (not destroyed) for 24 h as the rollback target (`run rollback <app>`), then destroyed.
7. **M, data backups.** **Fix:** the labs service snapshots each kept machine's `data` volume daily (tar, keep 7) under
   `/var/lib/ghost-labs/backups/`, plus before every redeploy/delete. `lab backup`/`restore` actions. Backups count against disk
   (finding 10).
8. **M, the model will try to reach a lab from the workspace or the browser** (psql/curl to a lab, a screenshot of the site). The sandbox
   and browser are Tor-only and cannot reach the tailnet. **Fix:** (a) a deterministic `execute` hint when a command names a
   `lab_` id or the configured labs host ("labs are reached with `lab exec`/`lab http`, not from the workspace"), and the same in
   the browser's refusal; (b) **visual check:** a `ghost-lab/browser` image (headless Chromium) and a `lab screenshot url=…`
   action that the service runs from a helper member on the lab network, returning a PNG to the workspace. The model
   gets the same "look at it" ability it has for sandbox previews.
9. **M, secrets.** Apps need DB passwords/session keys (no keyed external APIs, per the identity rule). **Fix:** the labs
   service generates them per kept machine (`secrets: [DB_PASSWORD, FLASK_SECRET]` in the recipe), stores them in its
   sqlite, injects them as env, and returns names only (never values). The agent redaction (`distill/redact.py`) also redacts the labs key.
10. **M, disk.** RAM is budgeted, disk is not (no-swap host, 101 GB free). **Fix:** `/v1/health` reports `disk_free_mb`. Creates
    are refused under a floor (e.g. 10 GB), `gc` prunes dangling images/build cache, and backups are capped.
11. **M, owner observability.** **Fix:** (a) `lab list` / the inventory shows RAM per machine and the totals vs budget; (b)
    an app's `crashloop` or kept machine down → one owner notification through the existing notify path (not repeated
    while still down); (c) `ghost-labs status` on the host for the operator.
12. **M, labs host down.** **Fix:** `labs_unreachable` → autopilot puts lab tasks on HOLD `labs_down` (auto-lifting, like
    `lab_capacity`). The briefing shows kept apps as `unknown (labs host unreachable)`, never as "up". Nothing falls back to the
    workspace.
13. **M, two owner conversations, one destroys the other's lab.** D7 allows it. **Fix:** destroying a lab owned by a **different**
    owner key (another conversation/project) needs the explicit `lab_id`, not a role or description, and the result names the owner
    it belonged to. The other conversation's next briefing shows "lab_x destroyed by <owner> at T".
14. **M, scheduled work inside apps** (a nightly job of the web+pg+worker case). `manage_tasks` schedules agent prompts, not
    machine cron. **Fix:** the base image includes `cron`, started by `start.d`. The recipe may declare `cron:` lines. Use
    `manage_tasks` only when the nightly work needs the agent (it then gets an owner, §3.4).
15. **L, naming for the model.** It sees exactly two nouns, **`lab`** (a machine; `keep: true` = kept) and **`run`**, with
    `lifetime: job | server | app` (renamed from task/preview/app, because "task" collides with project tasks). Docs say
    "machine" only to explain a lab. `where` is required on `run start` (unchanged). `app` requires a kept lab.
16. **L, previews stay on the workspace's 8100-8104 loopback range** (same launcher). Reaching a preview from another device still
    needs `serve-remote.sh`. Acceptable: previews are for the agent's own look, and the owner gets apps on the tailnet.

### Phasing (supersedes §8 ordering)
**A (host, incl. §6 + findings 4–11 host parts + §11 host rules) ∥ B0 (binding + finding 1–2) → B (`lab` tool, owners, §11
agent rules, finding 8a) → C (`run` built ALONGSIDE services; workspace launcher, lab adapter, promotion → `job` runs;
`is_promoted_result` kept) → D (projects on the inventory, release/deploy/rehearsal/versions/rollback per findings 4–6,
migrate chess-coach + jiu-jitsu, THEN switch tools: `run` replaces `manage_services`+`jobs`, the token check moves, the services
module is removed) → E (`postgres_admin` + Mac Postgres removal, kept-machine export/import, docs).**

---

## 13. Independent review r2 (2026-11-03): findings and fixes

r2 was a read-only reviewer who did not write the design. Its #1, #3 and #7 were re-verified by the author (`eva` = 100.64.0.12
via `dscacheutil`, the launcher's `--image-gen-nodes 100.64.0.13:8000|Ghost`; `create_version` forks a new project id at
`tools/projects.py:5436-5444`; `run_leaf_turn` calls `handle_chat` with no `session_id` at `core/coding_loop.py:490-493`).
Each fix below **overrides** earlier text.

### HIGH
1. **The D3 target was the image node.** Fixed in §5/§11: the agent is `eva`. Phase A test: the proxy reaches only the
   `--labs-agent-api` address, and nothing else on the tailnet.
2. **Docker can't supervise an app** (the contract entrypoint ends in `sleep infinity`). **Fix:** the manager runs each `app` as a
   supervised process in its machine. It records the pid via `exec detach`, and its health probe (HTTP/TCP, 60 s) **restarts it with
   backoff and stops it after 5 crashes in 10 min (`crashloop`)**. After a reboot, the manager's startup reconcile restarts
   every `app` of every running kept machine. D2 holds, enforced by the manager, not Docker.
3. **Versions fork a project.** **Fix:** the port lease and the data volume are keyed by **lineage + app name**. The lineage is the root
   project id, followed through `parent_project_id`. They are not keyed by lab or by project. A switch to v2 (a) snapshots `/data`,
   (b) stops v1's app, (c) attaches the **same** data volume to v2 (no copy), (d) moves the port after v2's probe passes. A rollback
   = start v1 against the same volume, plus an explicit choice: **keep current data** (default) or **restore the pre-switch
   snapshot**, offered in the result. An app that must not share data across versions declares `data: fresh`.
4. **The contract lacks every addition.** **Fix:** `docs/labs/ghost-labs-contract-v2.md` (written 11-03) is the Phase A
   spec: endpoints, fields, error codes and behaviour changes. It overrides the v1 prompt where they differ. Kept machines are
   exempt from the reaper, reconcile-destroy and `destroy --all`.
5. **The narrow path exposed the whole agent API.** **Fix:** the per-machine `agent-api` sidecar is an **HTTP reverse proxy** on the
   lab network. It allows only `GET|POST /api/game/*`, passes only `X-Ghost-Service-Token` and strips `X-Ghost-Key`,
   `X-Ghost-Requester`, `X-Ghost-Origin`, `Cookie` and `Authorization`, with a body cap of 64 KB. Its upstream is the address the
   agent sent. It runs on the host network side of the labs service, never inside the member. Tests: `/api/chat` → 403 through
   it; the master key header is stripped; another tailnet host is unreachable.
6. **Phase C would orphan promoted jobs.** **Fix:** execute's promotion switches to `job` runs **in the same deploy as the
   tool switch** (end of D). `run` results carry the same "still running, not evidence" marker. `is_promoted_result` keeps its name
   and semantics. Importers (floor): `tools/acquired_skills.py:958`, `tools/composed_skills.py:199,346`, `tools/registry.py:97`,
   `core/code_claims.py`, `core/build_gates.py`, `core/project_advancer.py:3079,3156`, `distill/outcome_heuristics.py`.
7. **B0 breaks writers and leaves.** **Fix:**
   - `current_project_id` gets a **compatibility setter** that writes the request's scope. There are 7 direct src writers, and 277 test
     assignments across 116 files keep working.
   - `handle_chat` accepts a pre-built `RequestScope`. `run_leaf_turn` passes its pinned scope, which replaces `_leaf_pinned_project`.
   - **No `session_id` → a request-only scope** (no fingerprint fallback). The leaf's after-turn check reads the scope object it
     passed in, not a ContextVar from inside the awaited task.
8. **Kept things owned by a conversation never die.** **Fix:** `keep` and `app` **require a project owner**. "Make it
   permanent" in an ad-hoc chat creates the project first (one call, which the tool does itself, titled from the lab's purpose) and
   moves the lab's ownership to it.

### MED
9. **Other global state.** The scratchpad `active_namespace` + snapshot/hydrate (`projects.py:525-533`) and
   `workspace_model.current_project_id` (`agent.py:31123`) are set from `RequestScope` on **every** turn. Hydrate only when
   **that conversation's** focus changes, not when consecutive turns come from different conversations.
10. **Tool lists for `run`.** Wherever `manage_services` or `jobs` appears, `run` takes its place:
    - the isolation denylist (`isolation.py:893`), `egress.CODE_TOOLS` (`:384`), `strikes.py:97`, `failure_replay.py:58`;
    - `agent.py:591, 768, 4043, 9698, 9883`;
    - add `run` to `OUTBOUND_TOOLS` (`url_literals`) and to `MACRO_FORBIDDEN_STEP_TOOLS` (`tools/composed_skills.py:244`; PLAN said
      `skills/`, which was wrong).
    Re-grep at D start.
11. **One teardown.** `teardown_inventory(owner, mode)` is used by the tool paths **and** the HTTP routes
    (`api/projects_routes.py:208` PATCH status, `:228-262` DELETE). A failed destroy goes to a `pending_teardown` ledger, which the agent's
    reaper loop retries (and reports after 3 failures).
12. **Capacity.**
    - `max_labs` counts **disposable** labs only.
    - The kept budget check includes the version-switch overlap (old and new up at once) and the rehearsal twin. A refusal gives the numbers.
    - The 1.5 GB kept budget fits one or two small apps. The web+pg+worker case alone is about 1.5 GB, so it would need D4 raised (ask then; don't assume).
13. **Backups and secrets.**
    - Backups go through a recipe `backup:` hook (e.g. `pg_dump` to `/data/backup.sql`), else stop-snapshot-start. A raw tar of live pg is
      never taken.
    - The host **redacts known secret values** from exec/logs/files output before returning it.
14. **Observability.** An agent-side poller (in `main.py`'s reaper loop, every 5 min) reads `/v1/health` + the kept-machine
    states. It notifies once per state change (dedup state on disk).
    - The briefing's RUNNING block uses a cached probe (≤60 s old, 1.5 s timeout). On timeout it shows `unknown (labs host
      slow/unreachable)`, so a slow host never stalls a turn.
15. **The manager is a single point.** At manager start, reconcile **re-applies member routes and forwarders** for every running
    member before serving. Forwarders run as per-port systemd units (`ghost-labs-fwd@<port>`), so a manager restart doesn't drop apps.
16. **Host identity.** `/v1/health` returns `host_id` (generated at install). Every inventory row stores it, and a mismatch
    shows `missing (host changed)` (§11). Export keeps `lab_id`, the port lease, secrets and backups.
17. **(c)/(i) details.**
    - The briefing shows **both** the focused project's inventory **and** this conversation's own ad-hoc items.
    - **Probe, member and replay requests never read or write focus rows**, even with a `session_id`.
    - Two web tabs that share a `localStorage` session id share a focus (documented; acceptable).

### LOW
18. **Naming and drift.**
    - Lifetimes are `job | server | app` everywhere.
    - **One way to make a kept machine:** `lab keep` (the `release` flow calls it). There is no `keep: true` on create.
    - The `lab` action list adds `screenshot, backup, restore`, and `run` adds `rollback`.
    - PLAN §4-§6/§9 items about `exec detach`, the `job` action and "archive destroys labs" are superseded by this doc (marked in PLAN).
19. **One conversation key.** Reuse the existing session-id helper (`agent.py:11928-11941`) and `conversation_key_context`
    (`utils/logging.py:35`). The other `conversation_fingerprint` users (`agent.py:19913, 21476, 23271, 30715, 33404`) move to it in B0.
20. **List corrections.**
    - Only **8 src files import** `sandbox.services`; the others mention it in comments only.
    - Browser loopback admission flows from `registry.py:1513` `active_service_ports`, which `server` runs must feed.
    - §7 also deletes `tools/delegate.py` (the `jobs` tool) and `tools/sandbox_services.py`.
    - `app.js:2263` is the tab id, not the session id.
    - `slack_project_commands.py` has a test (`tests/test_slack_project_commands.py`), which is deleted with it.
    - No interface/ client calls `/api/projects`, so r1 #1 is API-only (MED).
21. **Ready-to-start basics:**
    - §14 adds per-phase acceptance, deploy and rollback.
    - **Autopilot creates disposable machines only, and only when a task needs one** (operator 11-03). It never creates kept machines
      or `app` runs, and never runs `keep`/`deploy`/`switch`/`rehearse`/`rollback`/`restore`. Promoting to production is an owner request.
      Enforced agent-side (the autopilot leaf's calls for those actions are refused) **and** host-side (`origin=autopilot` → `403 owner_only`).
    - The `manage_services` → `run` alias maps args: `start name command port` → `run start lifetime=server where=workspace`; `status` →
      `status`; `logs lines` → `logs`; `stop-all` → refused with a hint.

---

## 14. Phases: acceptance, deploy, rollback

The suite runs once **per deployable phase** (each phase ships on its own: `scripts/run_suite.py -n 8` + lint). Every phase gets an
independent review before deploy. Deploy waits for `foreground_requests == 0`, then `launchctl kill SIGTERM` (the agent).

| Phase | Done when (acceptance) | Deploy | Rollback |
|---|---|---|---|
| **A** host | contract-v2 integration tests green on the labs host, including every PLAN §3b row as a data-reading test, the agent-api proxy tests (r2 #5), reboot survival (routes, forwarders, apps restarted), crashloop stop, and kept machines surviving the reaper | `install.sh` + systemd | `systemctl stop ghost-labs`; `ghost-labs destroy --all-disposable`; the host stays as-is |
| **B0** binding | two sessions on one project switch independently; a leaf keeps its pin; probe/member never touch rows; the console routes take `session_id`; the live global binding is dropped (B4) and the next owner message starts unfocused | agent restart | revert the commit + restore `scratchpad.db` from the pre-deploy copy |
| **B** lab | live probe (`X-Ghost-Origin`): create pg-single → `select version()` → destroy; an unconfigured host hides the tool; a down host shows `labs_unreachable`; the guard test (no hard-coded host) passes | agent restart + `GHOST_LABS_HOST`/key in the launcher | unset `GHOST_LABS_HOST` (the tool hides) |
| **C** run (alongside) | `run` starts/stops `server` runs in the workspace and `job`/`app` runs on labs; old services/jobs untouched; both registries agree in a reconcile test | agent restart | revert; the old tools never changed |
| **D** projects + switch | jiu-jitsu + chess-coach live on kept machines at stable tailnet URLs, data intact (row counts and saves match the pre-migration copy); a release rehearsal keeps data; v2 switch + rollback tested; then `run` replaces `manage_services`/`jobs`, promotion goes to `job` runs, the token check moves; the old modules are deleted | agent restart; apps cut over one at a time | before deleting the modules: re-enable the old tools (kept until D's live check passes); apps: start the sandbox copies again (the files never moved) |
| **E** cleanup | `postgres_admin` gone (PLAN §8 list + re-grep empty), the Mac Postgres removed (PLAN §10.4), export/import tested on a scratch kept machine, docs updated | agent restart | the `pg_dumpall` backup + `brew install` |

---

## 15. How apps reach the owner (added 2026-11-03, operator: "right now i have no easy way reaching a web application without a middle step")

### 15.1 Today (facts)
- Sandbox apps are published by Docker on **eva's loopback** (127.0.0.1:8100-8104). The owner reaches them only after running
  `~/Data/AI/bin/serve-remote.sh <port>` on eva, which runs `tailscale serve` and gives `https://mymac.tailnet-example.ts.net:<port>/`. That is the middle step.
  `tailscale serve status` on eva shows 8100, 3000 and 8088 mapped by hand.
- The tailnet already has **MagicDNS** and **HTTPS certificates** (eva has `mymac.tailnet-example.ts.net` certs; `serve` issues them).
- The labs host has Tailscale 1.104.5, `disorder.tailnet-example.ts.net`, and `tailscale serve` with `--https`, `--http`, `--tcp`
  and `--service` (Tailscale Services: a distinct name + virtual IP per service).

### 15.2 Design: the agent and the labs service do the exposing, never the owner
| Layer | What the owner gets | How | Setup |
|---|---|---|---|
| **1. Every app gets an HTTPS tailnet URL automatically** | `https://<labs-host>.<tailnet>.ts.net:<port>/` (today `https://disorder.tailnet-example.ts.net:20003/`), a valid certificate, works from any device signed into Tailscale | The forwarder binds **127.0.0.1:<port>** on the labs host (not the tailnet IP), and the labs service runs `tailscale serve --bg --https=<port> http://127.0.0.1:<port>` for it, and `serve --https=<port> off` on stop/destroy. Port = the stable per-(lineage, app) lease | none |
| **2. One page lists every app** | **`https://<labs-host>.<tailnet>.ts.net/`** (port 443), a read-only **Apps** page: name, project, state (up/restarting/crashloop/stopped), link, RAM, last deploy. One bookmark for everything | Served by the labs service (static HTML from `/v1/labs`, no secrets, no actions), via `tailscale serve --https=443` | none |
| **3. Previews are exposed too** | `https://<agent-host>.<tailnet>.ts.net:<8100-8104>/` for `server` runs in the workspace, with no `serve-remote.sh` | When a `server` run starts on a published port, the agent runs the same thing `serve-remote.sh` does (and removes it on stop). The Apps page also lists live previews (the agent pushes the list to the labs service) | verify that the launchd daemon can call the Tailscale CLI on macOS (`serve-remote.sh` already locates it in the app bundle); else a tiny root helper |
| **4. Optional: a name per app** | `https://chess.<tailnet>.ts.net/`, no port numbers | Tailscale Services (`tailscale serve --service=svc:chess …`). It needs a one-time admin-console step: define the services or a tag, and `autoApprovers` for the labs host | **one-time admin console + a test on day 1**; if it works, apps get names and layer 1 stays as a fallback |
| **5. Optional, per app: the home LAN** | `http://192.168.1.101:<port>/` for devices without Tailscale (TV, a guest) | The forwarder also binds the LAN IP. **Off by default** (the apps have no login); `lab keep … lan: true` turns it on for one app | operator choice per app |

- **Every reply that starts or deploys an app ends with its link** (from the labs API, never composed, per §11). The RUNNING block
  shows the same links. The web console gets an "Apps" link to the page of layer 2.
- **Notifications:** a deploy that finishes, or an app that goes down or comes back, sends one owner notification with the link (the §13 #14 poller).
- **Contract v2 deltas:** forwarders bind `127.0.0.1` plus optional LAN, and the tailnet side is `tailscale serve`. Add
  `GET /` (the Apps page) and a `previews` push endpoint `PUT /v1/previews` (agent → list of `{name, url, project}`). The URL fields
  return the `https://<host>.<tailnet>.ts.net:<port>/` form (derived at runtime from `tailscale status --json` → `Self.DNSName`).
- **Removed:** `serve-remote.sh` as an owner step (kept as a manual tool).

### 15.3 Decisions (exposure)
| # | Question | Recommendation |
|---|---|---|
| X1 | Layers 1–3 (automatic HTTPS links, the Apps page, previews exposed) | **Yes** |
| X2 | Try per-app names (Tailscale Services) on day 1 (a one-time admin-console step) | **Yes, try it**; fall back to ports |
| X3 | Allow LAN exposure per app (off by default) | **Yes, opt-in per app** |
| X4 | Which devices do you open apps on (phone, laptop, TV)? Do they all have Tailscale? | (owner's answer decides whether X3 matters) |

**Operator 11-03: X1 YES, X2 YES (try Tailscale Services on day 1), X3 YES (LAN opt-in per app), X4 "all kinds of devices but most
use tailscale".** So the tailnet HTTPS link is the default, and LAN stays opt-in per app for the devices without Tailscale.
**Day-1 task for X2:** the operator does the one-time admin-console step (define the service or a tag + `autoApprovers` for the labs
host); then test `tailscale serve --service=svc:<name> --https=443 http://127.0.0.1:<port>` from the labs host. Success → apps get
`https://<app>.<tailnet>.ts.net/`; failure → ports only, noted here.

### 15.4 The owner's Tailscale admin-console steps for X2 (from the Tailscale Services docs, kb/1552, read 2026-11-04)
Facts from the docs:
- A Service host must have a **tag-based identity**; a user-owned device can't host a Service.
- Each Service is **defined in the console** first. The docs show no API create, so the agent can't add names by itself, which also fits the
  no-keyed-API rule.
- Approval can be automatic through `autoApprovers` by **service tag**.
- Requires v1.86+ (the labs host has 1.104.5); clients need 1.94+ (older Linux clients need `--accept-routes`).

One-time:
1. Policy file: `tagOwners` for `tag:labs` (host) and `tag:labs-app` (services); `autoApprovers.services: {"tag:labs-app": ["tag:labs"]}`;
   a grant `autogroup:member → svc tag:labs-app, ip 443`; and, if the policy is not allow-all, a grant `autogroup:member → tag:labs`
   for 22, 8700, 443, 20000-20999. Tagging changes the device's identity, so per-user rules stop matching it.
2. Machines → `disorder` → Edit ACL tags → `tag:labs` (the device becomes tailnet-owned and its key expiry is disabled).
3. Define a test Service `labs-test` (endpoint `tcp:443`, tag `tag:labs-app`) for the day-1 test.

Per app, only when the owner wants a name: define a Service `<name>` with `tcp:443` + `tag:labs-app`. The agent says the exact name
to use. Approval is automatic. Without it, the app keeps its port link.

### 15.5 Day-1 result for X2 (2026-11-04): **per-app names WORK**
- The owner applied the §15.4 policy, tagged the labs host `tag:labs` (the first attempt did not save; on the second the netmap showed
  `Tags ['tag:labs']`) and defined `labs-test` (`tcp:443`, `tag:labs-app`).
- `sudo tailscale serve --service=svc:labs-test --https=443 --yes 127.0.0.1:18080` on the labs host was auto-approved (CapMap
  `service-host: svc:labs-test → 100.64.0.14`). From eva, `https://labs-test.tailnet-example.ts.net/` → **HTTP 200 with a valid certificate**.
- **Propagation delay:** the first request about 10 s after advertising timed out, and it worked a minute later. The service must
  probe the named URL with retries (≤ 2 min) before reporting a name as live, and report the port URL in the meantime.
- Needs **sudo** (`tailscale serve` as root). The labs service runs as root, so this is fine.
- Still unknown: whether the `tag:labs-app` grant alone covers new Services (the test also had the explicit `svc:labs-test` grant). **Check it
  on the first real app**: define it with the tag only, no policy edit. If it isn't reachable, add one `svc:<name>` grant line (owner step).
- Cleanup: the serve config was removed and the test server stopped. The `labs-test` Service definition stays in the console (harmless,
  reusable for tests).
- Unrelated finding: the labs host's netmap carries 10.x subnet routes from other tailnet routers (10.3/16, 10.4/16, 10.23/16, 10.24/16,
  10.100/15, 10.112/16, 10.120.132-135/24, 10.130.x, 10.132.x). The lab pool `10.89.0.0/16` does not collide. Pick any future pool
  outside those.
- **Contract v2 §12: `ts_services: true` from day 1.**
