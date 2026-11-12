# Phase D — projects on the inventory, kept machines, the switch (design, 2026-11-04)

Source of truth: UNIFIED_DESIGN.md §3.2–§3.4, §4, §5, §12 findings 3–7/11–13, §14 row D; PHASE_C.md for what C built.
**Operator change (11-04):** "i don't care about my current live apps. you can delete them." → there is NO migration of
chess-coach (48e0373aaab3) or jiu-jitsu-calendar (9009ff89d8bb); they are backed up and deleted. §14 D's acceptance
("data intact, row counts match") is replaced by: backup taken, both projects + their services + workspaces gone, nothing
else touched.

Code facts (sweep 11-04) are cited inline as `file:line`.

## D0. Order (each step ships with tests + docs + review + battery; suite once per deploy; deploy gated)
1. **D2 inventory** (projects own their runs/labs; one teardown for tool + console) → its live acceptance IS
2. **D1 delete the two apps** through that teardown (backup first).
3. **D3 kept machines + `app` runs + release** (host kept API already built in Phase A).
4. **D4 autopilot on labs** (task kind, leaf allowlist, holds).
5. **D5 the switch** (advertise `run`, aliases, promotion → run jobs, token check move, briefing RUNNING block, remove
   manage_services/jobs + the probe route, the 6 legacy service rows dropped).

## D2. Projects on the inventory
Today (`tools/projects.py`): delete (`:3426-3486`) and archive (`:3488-3512`) stop the project's **services** only
(`_stop_project_services`, `:2313`), and the console route `DELETE /api/projects/{pid}` (`api/projects_routes.py:235-271`)
is a SECOND, shorter copy (skips the human-gate refusal, `_forget_project_episodes`, `_remove_project_changelog`).
Runs (`runs.db`), labs (`lab_inventory.db` + host), sandbox jobs and schedules are not touched by either.

- **Inventory** `memory/inventory.py::inventory(ctx, owner)` → `{services, runs, labs, jobs}` for `owner=project:<id>`:
  services = `_project_service_entries`; runs = `RunRegistry.rows(owner, live_only=True)`; labs = the lab inventory rows of
  that owner (+ the host's `GET /v1/labs?owner=` when configured; a down host → `labs: unknown`, never "none");
  jobs = sandbox job rows with `project_id == pid` and state running. Read-only, no probing of labs (r1 M4).
- **One teardown** `tools/projects.py::_teardown_project(ctx, pid, *, hard)` used by the tool AND the console route
  (the route keeps its own gate checks, but the work is shared — the route's missing episode/changelog steps were a bug):
  - stop live **runs** of the owner through `run stop` (workspace and lab; a lab run's stop is a host DELETE);
  - stop **services** as today (`_stop_project_services`, incl. the legacy workdir fallback);
  - cancel the project's running **sandbox jobs**;
  - **labs**: archive destroys the owner's **disposable** labs and leaves **kept** ones up (named in the summary);
    delete destroys disposable AND kept (`include_kept=1`, `drop_data=1` — host takes the final backup itself, finding 4/7);
  - then the existing store/graph/episode/changelog steps.
  - Each step reports what it did; a failed step is named in the result and does NOT stop the others; the project row is
    deleted only if no step left a LIVE thing it could not stop (else: refuse with the list — never a row-less orphan).
- **Confirm preview** (`_lifecycle_confirmation`, `:2055`) lists the inventory (what will be stopped/destroyed), so the owner
  confirms the real blast radius.
- **status** (`:3570`) gains one `INVENTORY` block (counts + names, probed state only for workspace runs).
- Not in D2: schedules (no owner field today — `tools/tasks.py:103-120`; `scheduled_tasks.json` is empty) → D4.

## D1. Delete chess-coach + jiu-jitsu-calendar (live acceptance of D2)
Backup to `~/Data/AI/Data/backups/4sx-d-apps-<ts>/`: `sqlite3 .backup` of `system/memory/projects.db`; `tar` of both
workspaces (`sandbox/projects/<id>`); copy of `sandbox/.services/registry.json`. Then delete each through the console route
(live agent; single-writer stores, `memory/store_lock.py:40`) with `hard=true`. Verify: rows + tombstones, folders gone,
the 3 service rows (`48e0373aaab3:chess-coach-v3`, `9009ff89d8bb:jiu-jitsu-calendar`, legacy `jiu-jitsu-calendar`) gone and
their processes stopped, nothing else in projects.db/registry changed (diff vs backup).

## D3. Kept machines + `app` runs + release
Host (Phase A, owner-only `kept_call`, `interface/ghost-labs/ghost_labs/api.py:177-224`): `/keep`, `/deploy`, `/rehearse`,
`/switch`, `/rollback`, `/backups`, `/restore`, `/app/{start|stop|restart}`, `/secrets`; ports leased per `lineage/app`
(20000-20999, stable URL).
- `lab` gains **owner-only** actions `keep` (POST /keep: app {name, member, port, cmd, health}, lineage, data, secrets names),
  `deploy` (tar of a WORKSPACE folder → `archive_b64`, requirements, data_seed), `rehearse`, `switch`, `rollback`,
  `backup`, `restore`. Autopilot/probe/member refused agent-side first (D5 of the design: autopilot never touches `app`).
- `run` `lifetime=app` = a pointer (backend `labs_app`, ref `lab_id/app`) to the kept lab's app; start/stop/restart →
  `/app/*`; status/url from the lab detail `app` view (state, url, restarts, last_error). `where` must be a KEPT lab.
- `manage_projects release` → build: keep + deploy + rehearse; `unrelease` → stop the app (machine kept until delete).
  Today's release (services + RELEASED status, `core/prompts.py:137-160`) is replaced at D5, not before.
- Owner notification on `crashloop`/kept down (finding 11b): a reconcile-time check, one notice per down period.
- Detail to settle in D3's own design pass (host contract read first): agent_api sidecar use (no current app needs it).

## D3-design (11-04, after reading the host contract; overrides D3 above where they differ)
Host facts (api.py:176-227, kept.py): every kept endpoint is `owner_only` (X-Labs-Origin must be `owner`; autopilot/probe
refused HOST-side). `deploy` writes INTO the lab (no new lab); a new version is a second kept lab of the same
(lineage, app) that comes up `staged`, then `switch from_lab=<live>` moves data + the stable port to it (auto-reverse on
failure); `rollback` moves back. Stable URL = port lease `<lineage>/<app>` in 20000-20999 → `app.url` in the lab detail.
`app.state` ∈ stopped|starting|up|restarting|crashloop|staged; crashloop after 5 crashes/600 s.

**A. `lab` tool owner-only actions** (in the schema for EVERY origin — L11; refused at dispatch unless the caller is the
owner, except the probe rule below):
- `keep` lab_id + `app` {name, member, port, cmd, health?, workdir?} + `lineage?` `data?` (shared|fresh) `secrets?`
  (NAMES only) `cron?` `backup?` → POST /keep. Then the agent records a `run` row: lifetime `app`, backend `labs_app`,
  ref `<lab_id>/<app>`, name = app name, owner = the lab's owner. The lab must belong to a project (host: owner `project:`).
- `deploy` lab_id + `folder` (a WORKSPACE path; the agent tars it, gz, ≤ 20 MB → `archive_b64`) + `requirements?` {pip,apt}
  + `data_seed?` → POST /deploy. Paths go through the workspace path guard (no `..`, no symlink escape).
- `rehearse`, `switch` (from_lab), `rollback` (restore_snapshot?), `backup` (POST /backups), `backups` (GET),
  `restore` (backup_id) → the host endpoints, results rendered (probe ok/not, url, state, rollback_until).
- `app` lab_id + `app_op` start|stop|restart → POST /app/{op}.
- `secrets`: NOT model-facing (values would land in args/last_config.json and transcripts). Values are set by the operator:
  `PUT /api/operator/labs/{lab_id}/secrets` (X-Ghost-Key, owner, body {secret_values}) — never logged, never stored.
- Origin: the agent refuses autopilot/unattended/leaf/member/replay first (kept_only_owner). A PROBE may run these only on a
  lab whose owner is a `probe_created` project; for those the agent sends `X-Labs-Origin: owner` (the host has no probe
  tier for kept) — the probe still never touches an owner project's lab. (Needed so prod functional tests can exercise
  kept machines without owner traffic; reviewed as a scoped elevation.)
- After `switch`/`rollback` the agent repoints the app run row to the lab that now serves (backend_ref).

**B. `run` with lifetime `app`** (backend `labs_app`): not started by `run start` (refused: "use `lab keep`"); `run status`
→ lab detail `app` view (state, url, restarts, last_error); `run stop|start|restart` → `/app/{op}` (owner rules as A);
`run url` → `app.url`; logs → `lab logs member=<app member>`. `_stop` for the D2 teardown: stop the app; a delete then
destroys the kept lab anyway (D2).

**C. release** (`manage_projects release`): if the project has a live `app` run → the rehearsal gate is `POST /rehearse` on
its lab (ok = twin probe ok AND app state up), the dossier records the app url + lab id + lineage. With no app run →
today's path unchanged until D5 (services restart / deliverables). `unrelease` → `/app/stop` on each app run (the
machine is kept until delete) + today's demotion.

**D. Down notice** (finding 11b): a background loop every 300 s (like `_reap_sandbox_jobs`) reads the lab detail of each
live app run; state crashloop, or stopped with `last_error` and not `stopped_by_owner` → ONE owner notice
(autonomous_activity SEVERITY_NOTIFY, `notify_tool` rate limit) per down period; the period ends when state is `up`
again. Down period kept on the run row (note field `down_since`), so a restart does not re-notify.

**E. Not in D3:** agent_api sidecar (`agent_api_url`), `cron`/`backup` hook authoring help, `lan`. Of these, the tool
forwards only `data`, `secrets`, `cron` and `backup` to the host's `keep` (`tools/labs.py` `_kept_action`). `lan` and
`agent_api_url` are NOT passed (corrected 2026-11-11, review §4UD F18: this line said they pass through): each opens
a network path (the home LAN with no login; the agent's API through a sidecar), so the owner sets them on the host
API directly.

### D3-design review r1 (independent reader, 11-04): 5 HIGH, 6 MED, 3 LOW — these OVERRIDE D3-design
- **H1** no `lineage` argument: the agent always sends lineage = the lab owner's project id; the HOST also refuses a
  lineage that a kept lab of ANOTHER owner already uses (secrets/data/port hijack via a second "version").
- **H2** no agent elevation. HOST probe tier: a kept action from origin `probe` is allowed only on a lab whose stored
  origin is `probe`; a probe-kept lab gets `retire_at` = keep + 24 h (the reaper removes it); destroy/drop_data of such a
  lab is allowed to the probe too. Owner labs stay owner_only.
- **H3** HOST: mutating `exec`/`put`/`member` on a KEPT lab need the owner (or a probe on its own probe lab) — autopilot
  (D4 leaf) never touches a production member.
- **H4** app run rows never END while their lab lives: states `running|app_stopped|crashloop|staged` (none in ENDED);
  `run start name=<app>` on an existing app row = `/app/start`; `_stop` has its own `labs_app` branch (`/app/stop`, row
  → `app_stopped`). The row ends (`stopped`) only when the kept lab is destroyed (lab destroy / D2 teardown).
- **H5** release gate = the app run is `up` on its serving lab (no `/rehearse` on the live lab: it stops production).
  `unrelease` does NOT stop the app (today's demotion only). `rehearse` stays an explicit owner action.
- **M6** app row ref = `<lineage>/<app>`; the serving lab is resolved at probe time (the owner's kept labs, the one whose
  app is `mounted`) — no repoint after switch/rollback; long calls (deploy/switch/rollback) use the turn's remaining
  time, and a no-answer says "unknown — check `lab status`".
- **M7** one app row per (lineage, app); a staged second version shows in `run status` as "next version lab_x staged".
- **M8** down notice: run registry gets a `meta` JSON column; down needs 2 consecutive polls; host no-answer = unknown
  (no notice); notified mark only when the record was written; ONE notice per sweep for all down apps.
- **M9** deploy tar on the agent host with `os.fwalk` (dir fds, no symlink follow) + `O_NOFOLLOW` opens; symlinks,
  devices, fifos skipped (listed); excluded `.git node_modules .venv __pycache__ .env*`; ≤ 20 MB raw.
- **M10** teardown of a probe project's kept labs works through H2's probe tier.
- **M11** new inputs go inside the existing `params` object (per-action shape); `app` action uses the existing `op`.
- **L13** the keep result names the secrets the operator must set (operator route `PUT /api/operator/labs/{id}/secrets`).
- **L14** keep on a `conversation:` lab → "create the lab inside a project first".

## D4. Autopilot on labs
- Task kind: no new kind; `build`/`model_run` tasks may use labs. `core/coding_loop.py:59` `LEAF_ALLOWED_TOOLS` →
  `("file_system", "execute", "lab", "run")` with the leaf's origin = autopilot (host refuses kept/app; agent refuses first).
- Holds (`core/task_wait.py`): `labs_down` and `lab_capacity` as auto-lifting `wait_code`s (like `child_failed`, `:68`),
  lifted when `/v1/health` answers with room; blurb in `_ADVANCE_STOP_BLURB` (`tools/projects.py:1213`).
- Schedules: optional `owner` on `manage_tasks` rows (`tools/tasks.py:103`), shown in the inventory, removed by delete.

## D5. The switch (one deploy, the owner's tool list changes ONCE)
- `run` advertised (registry TOOL_DEFINITIONS + dispatch); `manage_services` + `jobs` definitions removed; their names
  become **aliases** to `run` for one release cycle (`_canonicalise_tool_name`, `core/agent.py:13182`) with a one-line hint.
- Every tool-name set re-grepped and `run` wired (sweep list: `core/agent.py:365,591,775,4068,9876,10060`,
  `core/isolation.py:891,900`, `core/strikes.py:97`, `core/failure_replay.py:58`, `memory/egress.py:387,515`,
  `utils/provenance.py:50,108`); batch order (`_batch_call_depends`: run after execute in the workspace; run on lab_x after
  lab calls on lab_x); model-facing text (`core/prompts.py:144,239`, `tools/execute.py:60,780-800`, `tools/projects.py`,
  `tools/browser_support.py:342`, `core/build_gates.py:366,387`, `main.py:514,633,663`).
- Promotion: `execute`'s promoted job registers a `run` row (lifetime job, started_via "execute"); the banner names
  `run collect name=…`; `is_promoted_result` kept.
- Token check: `valid_service_token` (`sandbox/services.py:2832`) has one caller (`api/game_routes.py:74`) and no remaining
  user (chess-coach deleted in D1) → moved to the run registry (workspace server rows' tokens) — or removed with the
  game routes' token path if nothing uses it; decided in D5 review.
- Briefing: one RUNNING block in the dynamic state (`core/agent.py:32321-32340`) from the inventory (probed workspace runs,
  labs from the lab inventory, never a host call per turn).
- Removed: `tools/sandbox_services.py` tool surface, `tool_jobs`, the operator probe route. The supervisors stay as the
  run adapters (UNIFIED §7's module merge is E/out of scope — it would rewrite 2.9k lines for no owner-visible gain).
- The 6 legacy preview/test service rows: stopped + dropped from the registry (files untouched).

## Rollback
Each step reverts by redeploying the previous tree; D1 restores from the backup (projects.db + tar); D5 re-enables the old
tool definitions (the modules stay until E).

## D6. Design review r1 (11-04, independent reader) — 3 HIGH, 7 MED, 3 LOW; these OVERRIDE D0–D5
- **H1 origin:** the console route has no request scope → labs/runs default to `autopilot`. `_teardown_project(..., origin=)`
  takes the origin EXPLICITLY; the route maps it from `_request_origin`; a probe origin is refused on any project not
  `probe_created`. (Fact fix: the DELETE route has NO own gate checks; `_stop_project_services` is at `:2297`.)
- **H2 archive keeps production:** archive stops `job`/`server` runs + disposable labs only; `app` runs and kept labs stay
  up and are LISTED with a question in the summary (§3.4).
- **H3 lineage data:** delete sends `drop_data=1` only when NO other kept lab exists in the lineage; else destroys the lab,
  keeps the data and says so; the preview names the lineage (kept.py:1439-1449 drops volumes whenever mounted).
- **M4** `PATCH /api/projects/{pid}` status=ARCHIVED (projects_routes.py:214) goes through the same teardown.
- **M5** refuse only when the inventory has LIVE rows that could not be stopped; with the host/sandbox down and no rows
  → proceed with a note (no `pending_teardown` ledger — said plainly: not built).
- **M6** D4 gives the leaf `lab` only; `run` joins the leaf at D5 (it is not registered before).
- **M7** D5 sweep adds `egress.OUTBOUND_TOOLS` (:215, `"run": "url_literals"`) and `MACRO_FORBIDDEN_STEP_TOOLS`
  (composed_skills.py:244).
- **M8** the confirm plan stores an inventory digest; a "yes" whose inventory gained a kept/app/data item re-previews.
- **M9** RUNNING block: cached probe (≤60 s), absolute times, dynamic tail only; never a docker exec per turn.
- **M10** D1 rollback is BEST-EFFORT (operator does not care about the apps); acceptance compares the OTHER rows by id/content
  (excl. the two ids + `deleted_projects`), not a whole-file diff.
- **L11** the tool schema changes at D3 too (new `lab` actions) — one re-prefill per deploy; owner-only actions stay in the
  schema for every origin and are refused at dispatch (schema never varies by origin).
- **L12** aliases map ARGS too at dispatch (manage_services start → `run start lifetime=server where=workspace`; stop-all
  refused; jobs collect → `run collect`).
- **L13** D1 facts: nothing listens on 8100-8119 (rows are stale; D1 only removes rows); the legacy jiu-jitsu row shares
  8101 with `webos-server` — check `ss` right before and make sure the stop cannot kill another listener; send OWNER origin
  on the D1 calls; tombstones arm the 24 h delete-then-recreate note (`find_deleted_similar`).

## D5-design (11-04, after the sweep; overrides D5 where they differ)
- **Schema:** `run` advertised (`tools/runs.py` `RUN_TOOL_DEFINITION`; lifetime enum job|server — `app` comes from
  `lab keep`). `manage_services` + `jobs` definitions leave TOOL_DEFINITIONS; their CALLABLES stay registered as
  **alias functions** (not advertised) that map the arguments and prefix a one-line hint, so a model that remembers the
  old names still works for one release cycle:
  - `manage_services start name command port` → `run start where=workspace lifetime=server`; `stop|restart|logs` → the
    same on `run`; `status` → `run status [name]`; `stop-all` / `adopt` → refused with the `run` way to do it.
  - `jobs status` → `run status`; `jobs collect|cancel job_id` → the run row whose backend ref is that job
    (`collect` / `stop`); no such row → refused naming `run status`.
- **Wiring:** every name set that holds `lab` or `manage_services`/`jobs` is re-read and `run` placed by MEANING
  (mutating, verify, work, batch wall, drop note, replay-forbidden, macro-forbidden, egress code tool + url_literals,
  program output, evidence, provenance); `_batch_call_depends`: run after execute; run on `lab_x/…` after lab calls on
  lab_x; lab destroy after run on that lab.
- **Promotion:** `execute`'s promoted job is recorded as a `job` run (name `job-<8hex>` from the job id, started_via
  `execute`, the request's owner); the banner and the landed-job wake text name `run collect name=…`.
- **Token check:** stays — run's workspace servers ARE the services supervisor's entries, which mint and revoke the
  token; nothing to move (D5 review decision).
- **Briefing:** the SERVICES line becomes RUNNING: the project's live run rows with their last probed state + time and
  its live labs from the inventory (local records only; no probe/host call per turn — D6 M9).
- **Removed:** the operator probe route `/api/operator/run` (prod functional tests drive `run` through probe-header
  chat turns), `tool_jobs`/`tool_manage_services` from the schema. The 6 legacy service rows are stopped + dropped at
  deploy (operator step, backed up first).
- **Release:** a project's live workspace `server` runs join the rehearsal as its services did (the services path
  reads the supervisor entries, which run servers are — unchanged mechanics, named as runs in the dossier).
