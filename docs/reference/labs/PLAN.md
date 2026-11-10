# §4SX — ghost-labs: disposable lab containers on `disorder` (replaces `postgres_admin`)

Status: **PLANNED 2026-11-03, implementation starts 2026-11-04.** Nothing built yet.
**⚠ 2026-11-03 later: `docs/labs/UNIFIED_DESIGN.md` proposes unifying workspace/labs/services/jobs/projects. If the operator
approves it, its §8 phasing REPLACES phases B–D below; Phase A (host) is unchanged plus UNIFIED §6.**
Contract for the host service: `docs/labs/ghost-labs-manager-prompt.md` (written for a Pi; the
deltas in §3 below override it). Journal number: **§4SX** (the memory file says §4SW, but the
journal's §4SW is the Greek voice, so §4SW is taken).

---

## 1. Operator decisions (2026-11-03)

| # | Decision |
|---|---|
| D1 | Throw away `postgres_admin` entirely. Replace it with **general** disposable labs, not Postgres-only. |
| D2 | Host = **`disorder`**. It runs only Docker and the `ghost-labs` service. Claude builds it over ssh (sudo works), not Gemini. |
| D3 | Agent side minimal: one `lab` tool. The agent connects through the env var **`GHOST_LABS_HOST`**. The host service does the lifecycle work deterministically, with no LLM. |
| D4 | The agent has full root inside labs. Lab egress is **strictly Tor**: fail-closed, no clearnet path, ever. An offline lab (`egress: none`) is also allowed. |
| D5 | `disorder` itself reaches the internet **freely at setup time only** (apt, Docker Hub, PGDG, pip during `install.sh` / `build-images`). Labs never pull at create time. |
| D6 | A self-hosted site is reached through a **port on the tailnet** (`http://100.64.0.10:20xxx/`). No onion and no LAN/public exposure. |
| D7 | **The agent and autopilot may both create and destroy labs freely**, for ad-hoc requests and projects, with no owner confirmation. |
| D8 | A project / experiment may use **more than one lab**, and one lab may have many members (containers). |
| D9 | New kinds of labs (e.g. MySQL) come in as **data** (Dockerfile + template YAML) on `disorder`, never as agent code. Until a template exists, the agent installs things in a `blank-dev` lab over Tor. The agent may *propose* a recipe, and the operator approves before the image is built. |

## 2. `disorder` facts (checked 2026-11-03)

- Ubuntu 24.04.5 LTS, **aarch64**, 4 cores, 7.9 GB RAM, **no swap**, 117 GB disk (101 GB free).
- Docker 29.9.2 is active, cgroup v2, apparmor and seccomp default. The firewall backend is **iptables**, with the userland proxy on.
- Tailscale 100.64.0.10; LAN 192.168.1.101 (not used, because macOS blocks launchd daemons from LAN IPs, see memory `macos-daemon-local-network`). No ufw. NTP is synced. iptables policy is ACCEPT everywhere, with only Docker's chains.
- Nothing ghost-labs is installed. No containers and no images.
- `sudo -n` works.

## 3. Deltas from `ghost-labs-manager-prompt.md`

1. **OS:** Ubuntu 24.04 host (not Raspberry Pi OS / Debian trixie). The lab *images* stay Debian trixie-slim based.
2. **Egress (D4):** the default for every template is **`tor`**. `none` is allowed, and there is no third mode.
   **Transparent** Tor, not proxy env alone: the sandbox learned in §4FU that proxy env leaks (tools
   that ignore it went direct from the real IP, 161 commands). Design:
   - Each lab network stays `internal: true` (no gateway).
   - A per-lab or shared Tor gateway container sits on the lab network with `TransPort` + `DNSPort`, plus
     iptables in **its own netns** REDIRECTing all TCP from the lab subnet to TransPort and DNS to DNSPort,
     and DROPping everything else (UDP, ICMP out). The original destination survives because the NAT happens
     in the gateway's netns (the same reason as §4FU).
   - On member start, the manager (root on the host) sets the member's default route to the gateway's lab IP
     with `nsenter -t <pid> -n ip route replace default via <gw>`. Members do **not** get NET_ADMIN by default, so they
     cannot change it. With NET_ADMIN opted in, they could, but the network is internal, so there is still no other
     route out (fail-closed holds).
   - Keep `ALL_PROXY`/`HTTP(S)_PROXY` + apt proxy as a belt (some tools are faster with SOCKS).
   - **Fail closed:** if the gateway is not up, a `tor` lab create fails (`tor_unavailable`, 503). Never
     fall back to direct.
   - Test: from a member, a plain `curl https://check.torproject.org/api/ip` gives `IsTor:true`, and
     `curl --noproxy '*'` also gives IsTor true (transparent). A UDP query to 1.1.1.1:53 fails. A raw IP
     connection to a LAN host (192.168.0.x) fails.
   - Labs must not reach the LAN, tailnet or `disorder` host services. Tor's `ReachableAddresses` / an
     ExitPolicy reject on private ranges in the gateway's torrc, plus a test.
3. **Published ports (D6)** are bound on 100.64.0.10 only, through a **host-side forwarder** (proven in §3b) that
   listens on the tailnet IP and forwards into the lab network. Lab members never get a non-internal interface, and
   the contract's `-pub` network is NOT built. Forwarders die with their lab (part of destroy and of the leftover check).
4. **Boot ordering:** `docker.service` is not ordered after `tailscaled.service`, so binding to the
   tailnet IP at boot can fail. Add a systemd drop-in `After=tailscaled.service` + wait-for-IP for
   `ghost-labs.service` (and for Docker if Docker binds the ports itself).
5. **No swap:** `memswap_limit == mem_limit` (already in the contract). Also set the manager's own
   `OOMScoreAdjust=-900` so the OOM killer picks a lab, not the manager. Budget auto = 75% of 7.9 GB −
   1 GB ≈ **4.9 GB**, `max_labs: 4`.
6. **Who called (D7 / §7 below):** `POST /v1/labs` takes `origin` (`owner|autopilot|adhoc|probe`),
   `project_id`, `conversation_id`, and `purpose` (≤120 chars). These are stored as labels, returned in
   summaries, and written to `actions.log`.
7. **Pre-emption (§7):** an owner create that hits the budget evicts the oldest **autopilot** lab and
   reports it (`evicted: [{lab_id, project_id}]`). Autopilot never evicts; it gets `budget_exceeded`.
8. **Ad-hoc members:** `POST /v1/labs` may take `members: [{name, image, mem_mb, cpus}]` in place of
   `template`, with images from the built catalog only (no pulls). That gives "web + pg + redis" without a template
   per combination.
9. **Reachability probe:** `POST /v1/labs/{id}/http {member, port, path}` → status + headers + body
   head, fetched **from the host** over the lab network. The agent's sandbox and browser go through Tor and
   cannot reach the tailnet, so this is how the agent verifies "the site is up" from outside the container.
10. **Recipe export:** `GET /v1/labs/{id}/spec` → the exact create body (template/members/params), so a
    project can save it as its recipe.
11. **Destroyed-lab memory** also records `by` (origin) so "where did my lab go" has an answer.
12. **Reboot of disorder:** containers come back (`restart: unless-stopped`), and `start.d` scripts rerun, so
    services restart and data (named volumes) survives. Detached jobs that were running are marked `lost`, never rerun.
    TTL clocks continue from sqlite.
13. **`build-images` / `install.sh`** are the only things that use the clearnet (D5). After setup, block the
    host's own outbound traffic? **Not decided.** D5 says "freely at setup time only". Proposal: leave the host egress
    open (the labs are already fail-closed), and log every `build-images` run. → Ask the operator if a host-level
    lockdown is wanted.
14. **Templates shipped on day 1:** `blank-dev`, `pg-single`, `patroni-3`, `web`, **`mysql-single`**
    (proves the "add a kind" path, D9). Readiness for MySQL is `mysqladmin ping`.

## 3b. Egress prototype on `disorder` (2026-11-03, PROVEN by hand, then fully cleaned up)

Built from throwaway `labtest-*` containers. Results:

| Check (from a lab member, no proxy env, no NET_ADMIN) | Result |
|---|---|
| Internal network + gateway container (tor, TransPort 9040, DNSPort 5353, `nat PREROUTING -i <lab-if> tcp → 9040`, `udp 53 → 5353`, `FORWARD DROP`) + member default route via the gateway set with `nsenter -t <pid> -n ip route replace default via <gw>` | works |
| `apt-get install curl` with NO proxy configured | ✅ via Tor |
| `curl https://check.torproject.org/api/ip` | ✅ `IsTor: true` |
| Raw-IP TCP (1.1.1.1:80) | ✅ via Tor (transparent) |
| UDP 53 to any IP | answered **by Tor's DNSPort** (redirected, not a leak). Other UDP / ICMP: dropped |
| LAN (192.168.0.x), tailnet (100.x), `docker0` | ❌ unreachable (Tor refuses private addresses). NB: `/dev/tcp` connect "succeeds" because TransPort accepts first, so **tests must read data, not just connect** |
| Member names (Docker DNS) | ✅ |
| **External names via Docker DNS on an internal network** | ❌ **by default.** Fix PROVEN: create members with `--dns <gw-ip>`. The embedded DNS then forwards to the gateway, which redirects to Tor, and member names still resolve |
| **Host's bridge IP (172.18.0.1:22 → disorder's sshd)** | ⚠ **REACHABLE: a real hole.** On internal networks the host keeps the bridge gateway IP. Fix PROVEN: host `iptables -I INPUT -s <lab pool> -m conntrack ! --ctstate ESTABLISHED,RELATED -j DROP`. After it: banner gone, member DNS + Tor still work, host→member (forwarder) still works |
| Tailnet port publish: host `socat TCP-LISTEN:20000,bind=100.64.0.10,fork TCP:<member>:80` | ✅ from disorder AND from the Mac over the tailnet. LAN IP 192.168.1.101:20000 refused. No `-pub` network needed → **host-side forwarder is the design** (§3.3), with the manager owning the forwarders (asyncio or supervised processes: a socat started from an ssh session died with the session) |

**Requirements this adds to Phase A** (all new vs the contract):
1. `/etc/docker/daemon.json`: `default-address-pools` = a dedicated range (proposed `10.89.0.0/16`, /24 per network).
   The default pools include `192.168.0.0/16`, which **collides with the LAN 192.168.0.0/24**. Also `log-opts max-size`.
   Then ONE persisted host INPUT DROP rule for the whole pool (NEW connections from labs to the host), installed by
   `install.sh` and checked by the manager at start (refuse to start without it: fail-closed).
2. The gateway gets a **fixed IP** on each lab network (e.g. `.2`), because members' `--dns` points at it. ⚠ Docker refuses `--ip` on a
   pool-allocated subnet (verified 11-03), so the manager allocates explicit subnets with an `--ip-range` (contract v2 §11.1).
3. Members: `--dns <gw>`, `--dns-search lab.invalid` (Docker copied the host's tailnet search domain
   `tailnet-example.ts.net` into the lab's resolv.conf, a small leak), `sysctl net.ipv6.conf.all.disable_ipv6=1` (Tor DNS
   returns AAAA, and there is no v6 route, so apps should fall back to v4 fast).
4. **The default route is lost on every container (re)start** (member start/restart, `rejoin`, disorder reboot). The manager
   re-applies it from a Docker **events** listener (`start` event → nsenter route), and readiness waits for it. Without
   it the member is offline (fail-closed, but broken).
5. The gateway's OWN egress: it sits on a normal network and **can reach the LAN and the host** (tested). Harden: DOCKER-USER
   DROP from the gateway network to RFC1918 / 100.64/10 / 169.254/16, plus host INPUT DROP from that subnet. The agent
   never execs in the gateway (it is not a lab member, has no `ghost.lab.member` label, and the API refuses it).
6. Gateway: one **shared** `ghost-labs-tor` attached to each lab network (bootstrap once, about 30 s), `FORWARD DROP` so labs
   can't route through it to each other, `TransPort … IsolateClientAddr` (circuits per member). Tor data dir on a
   named volume so restarts bootstrap fast. A `tor` lab create waits for `Bootstrapped 100%`, else `tor_unavailable`.
7. Host packages installed at setup (D5): `socat` (installed 11-03 during the prototype), `tor` is NOT needed on the
   host (it runs in the gateway image). `debian:trixie-slim` was already pulled 11-03.
8. Ops lesson: `pkill -f <pattern>` over ssh matched the ssh command's own bash and killed the session. Kill by
   pid or unit, never by a pattern that appears in your own command line.

## 4. How the agent separates its sandbox from labs

- **Two tools, never a routing flag.** `execute` = the agent's own sandbox (workspace, project files,
  deliverables), unchanged. `lab` = the system under test. There is no "current lab" mode that silently redirects
  `execute`. Anthropic's code-execution docs warn about exactly this confusion when mixing environments
  ("multicomputer environment").
- Every `lab exec` names `lab_id` + `member`. Every result starts with a header line `[lab_3f2a/pg1 exit=0 1.2s]`.
- Rule in the prompt (one line): *you write things in the sandbox; you run the system under test in a lab;
  anything in a lab is disposable, so copy results back to the sandbox.*
- File movement is explicit: `lab put` (sandbox path → member path) and `lab get` (member path → sandbox path).
- Long-running servers go through `lab exec detach=true` (job id). The sandbox's `&`/daemonized-server blocks in
  `tools/execute.py` do **not** apply to `lab`.

## 5. Agent-side `lab` tool (`src/ghost_agent/tools/labs.py`, new)

Actions: `templates`, `create`, `list`, `status`, `exec` (+`detach`), `job` (get/kill), `put`, `get`,
`member` (stop/start/restart/kill/pause/unpause/isolate/rejoin), `logs`, `http`, `extend`, `destroy`,
`spec`.

- Config: `GHOST_LABS_HOST` (a runtime setting, see UNIFIED_DESIGN §11; today's value `100.64.0.10:8700`), key in `~/Data/AI/.ghost_labs_key` (0600,
  copied from `disorder:/etc/ghost-labs/api_key`). If either is unset, the tool is **unconfigured and hidden**,
  using the same mechanism `registry.py:1345` uses for `postgres_admin`/`default_db`.
- HTTP client: plain httpx/curl_cffi **direct**. 100.64/10 is already Tor-bypassed by `compute_tor_proxy`.
  Address by the tailnet IP, never LAN.
- Errors are passed through as the service's `message` + `hint` (tell the model what it lacks, §4KL). An
  `unknown_lab` for a destroyed lab says when and why, so the model recreates the lab and doesn't retry.
- **Result shape (review r1 #12):** return `ToolOutcome` (`tools/outcome.py:101`) with the status set, and the body MUST
  contain the literal `EXIT CODE: <n>` line. `_EXIT_CODE_RE` (`outcome.py:65`), `code_claims._exec_ok`,
  `claim_binding.py:~1280` and `project_advancer.py:~3120` all parse that, and a custom `[lab/pg1 exit=0]` header
  alone is invisible to them. Use the header PLUS `EXIT CODE:`. A timeout → `timed_out` stated plainly, with no exit code.
- **Output cap on the agent side (#14):** the service returns up to 256 KB per stream, against a shield limit of
  max(16 000, 10% of the char budget) (`agent.py:~24832`). Cap per stream on the agent side (keep the tail, e.g. ~12 KB, and say
  "truncated, N bytes; use `lab get` for the full log"). Do **not** add `lab` to the shield exemption list.
- **Long operations vs the turn deadline (no per-tool timeout map exists):** `create` of patroni-3 can take minutes
  (readiness up to 240 s). The tool clamps `wait`/`timeout_s` to the turn's remaining budget (the §4RA deadline). Past
  that it returns `state: creating` + "check with `lab status`", and does not block. Add `lab` to `_BATCH_WALL_TOOLS`
  (`agent.py:~7005`).
- **Config lives on `args`, not just env (#10):** `--labs-host` (default `$GHOST_LABS_HOST`) + `--labs-key-file`, so the
  isolated copy's capability clear (`isolation.py:~1233`, the `default_db` loop) can null it. Replace `default_db` in
  that loop with `labs_host`. Exported in the live launcher `~/Data/AI/bin/start-ghost-agent.sh` (outside the repo;
  also update its `--default-db` header comment, lines ~29-30).
- **Key redaction (#11):** `distill/redact.py:~571-590` redacts only the `GHOST_API_KEY` value. Add the labs key the same way.
- The tool schema is kept **small**. Tools render before the system prompt (§4LE/§4RC), so any schema byte change
  re-prefills the cached head once at deploy. One-time cost only, but don't put per-request content in it.
- Templates are discovered at run time (`templates` action / `GET /v1/templates`). The schema does not hard-code the list.

## 6. Wiring into projects

1. **The recipe is in the project; the lab is a cache.** Labs die (4 h idle, 24 h max), and projects last days. The
   project workspace keeps `labs/<role>.json` (from `GET …/spec`) + `labs/<role>.setup.sh` (what the agent
   did after create). "Resume tomorrow" = recreate from the recipe. The recipe is also part of a reproducible
   deliverable.
2. **Ownership:** a `lab create` inside a project auto-sends `project_id`. The project keeps a lab list
   (lab_id, role, template, origin, expires_at) in a small `project_labs` table or project metadata.
   **Shown in `build_project_briefing` (`core/prompts.py:44`)**, the dynamic-state block, which is KV-safe (review r1 #1:
   `pinned_project_context` does NOT render; it only pins `current_project_id`). Ad-hoc labs belong to the conversation
   (`conversation_id`) and are shown in the same dynamic block for that conversation.
3. **Lifecycle hooks:** `delete_project` (`memory/projects.py:615`) is a **synchronous store method** (#17). Do NOT
   call the network inside it. The tool-level callers (`tools/projects.py:~3784,3816`) read the project's lab list
   first, delete, then destroy the labs (async, best effort, logged). A failed destroy is still caught by the service
   reaper, since labs carry `ghost.lab.project=<id>`, and the service can destroy by project
   (`DELETE /v1/labs?project_id=` — add to the API). **Hard delete** destroys all of the project's labs. **Archive** destroys
   them and keeps the recipes. **Resume** recreates nothing until a task needs it.
4. **Task kind `lab`** (`core/task_kind.py:29` `TASK_KINDS`): DONE requires a kept **lab-exec transcript**
   (cmd, member, exit code, output tail), written by the `lab` tool to `<memory_dir>/lab_transcripts/<project>/`
   (deleted with the project, like probe transcripts). Built like §4ST `core/experiment_gate.py`.
   Also needs: `ROUTE_LABEL` (`task_kind.py:66`), the docstring (`:7-12`), the `task_kind` enum in the manage_projects
   schema (`tools/projects.py:~6160`), and a branch in the advancer's declared-kind routing (`project_advancer.py:~1550-1560`).
   **Experiment conflict, corrected (#3):** `is_model_experiment` holds only when a model is NAMED (or a sibling names
   `probe_model`), so "failover experiment on the patroni lab" is NOT held. The real risk is narrower: the planner
   declaring `kind="experiment"` (an alias → `model_run`) for a lab task. Fix by **declaration, not wording**: the
   planner prompt says to declare `lab` for lab work. **No new wording classifier** (`task_kind.py:4-5` records that
   wording rules took 5–11 rounds and flip-flopped; memory `fuzzy-judge-as-gate-does-not-converge`). Pin a test: a
   declared `lab` task with "experiment" in its text routes to `lab`, and its gate is the lab transcript, not §4ST.
5. **Claim ledger** (`core/code_claims.py`, §4PH) (#13): it binds a written sandbox FILE to a later `execute` that
   names it, and `CODE_EXTS` has no `.sql`. For labs: bind a sandbox file sent with `lab put` to a later `lab exec`
   whose cmd names the put path (the put→exec chain), and add `.sql` (+ `.sh`, `.yaml` for lab recipes) to the lab
   binding. A claim of a cluster behaviour with no file ("failover works") is bound to the lab transcript instead:
   the transcript is the evidence the `lab` gate checks. Do not try to make the ledger understand clusters.
6. **Capacity (#8):** `budget_exceeded`/`lab_limit` → HOLD with a **new `wait_code` = `lab_capacity`** that the advancer
   lifts automatically when `GET /v1/health` shows room (checked at most once per tick). Today only `child_failed`
   auto-lifts (`task_wait.py:17-21`), and an `owner_turn` hold would wait for the owner forever.
7. **Autopilot (D7) — CRITICAL (#6):** autopilot today can't reach `lab` at all. The advancer routes only to
   needs_user / research / coding / self_analysis / model_run-hold. The coding leaf runs with
   `LEAF_ALLOWED_TOOLS = ("file_system", "execute")` (`coding_loop.py:59,443,481`). Build a **lab route**: declared kind
   `lab` → `run_leaf_turn` with a lab allowlist `("file_system", "execute", "lab")` and the lab gate at close.
   Without this, D7 is false for projects. Every lab carries `origin` (§4PF).
8. **Origin plumbing:** the tool sends `origin` = `autopilot` when called from an advancer tick (the §4PF origin is
   already computed in `project_advancer.py:~1303-1345`), `owner` for owner requests, and `probe` when `X-Ghost-Origin` is set.

## 7. Autonomy guard rails (enforced by the service, not by the model)

- Autopilot may destroy labs it created. It may destroy any other lab only if no **owner** request touched it in
  the last 30 min (the service tracks `last_owner_use_at`).
- Owner work pre-empts autopilot capacity (§3.7). Autopilot waits.
- Every create/destroy goes to `actions.log` with origin, project, conversation and purpose. The agent logs one
  `pretty_log` line per create/destroy with a distinct Icon (memory `logging-convention`).
- Probe/test traffic (`X-Ghost-Origin`, memory `probe-header-or-it-is-user-traffic`) creates labs with
  `origin=probe` and a short TTL.

## 8. Other places that must change (found in the 2026-11-03 scan)

**Remove `postgres_admin`:** `tools/database.py` (485 lines, deleted), `tools/registry.py` (import `:4`, schema
`:739`, unconfigured check `:1345-1348`, dispatch `:1574`, docstring `:1037-1043`), `core/prompts.py:945`
(DATABASES line, which also points the model at the default DB through execute/psycopg2 → replace with the one-line
LABS rule) + `:920` (advertises psycopg2 in the sandbox: keep, but drop any "default DB" wording), `core/agent.py`
(`:2454` is `_EXTERNAL_EVIDENCE_TOOLS`: swap `postgres_admin` → `lab`; aliases `:12981-12983` → map
`postgres`/`postgresadmin`/`postgres-admin` to `lab`, since a model reaching for "postgres" now means a lab; shield list
`:24833`: remove `postgres_admin`, do NOT add `lab`),
`core/metacog.py:66`, `core/turn_state_check.py:644` (`_LOCAL_READ_TOOLS`), `tools/fallback_chains.py:32`,
`tools/tool_failure.py:630`, `tools/validators.py:493,566` (destructive-DDL validator for postgres_admin, drop
it), `tools/execute.py:1993`, `core/isolation.py:893` (denylist) + `:1224-1233` (`default_db` capability clear),
`main.py:343` (`--default-db` / `GHOST_DEFAULT_DB`), `eval/behavioral.py:40-90` (`db_scalar` grounded DB
check → retarget to a lab or drop those cases, including its caller at `:235`), `scripts/tool_probes_live.py:50`,
`scripts/ablation_eval.py:124` (also uses the local DB). Tests: **32 test files**. Leave stale history alone (live
`foresight/gate.json`, `foresight/predictions.jsonl`, the 11-01 census report, and `system/last_config.json` records
`default_db`; that's history, not config). Sandbox `postgresql-client-18` (§4SF, `sandbox/docker.py:123`): **keep**
(psql in the sandbox talks to lab DBs only through the lab tool's exec, so it is harmless; removing it means a sandbox
image rebuild for no gain). Nothing in src deliberately uses `host.docker.internal:5432` (`validators.py:586`
mentions it), so it can go with the validator text.

**Other tool lists that need `lab` (review r1 #9; each verified by grep at implementation time):**
- `memory/egress.py:384` `CODE_TOOLS` (§4MB outside-content refusal), as well as `OUTBOUND_TOOLS`.
- `core/strikes.py:97` `PROGRAM_OUTPUT_TOOLS` and `:424` `READWRITE_LOOP_TOOLS`, otherwise status→exec→status loops get
  force-stopped.
- `core/agent.py:767` `_EDIT_CHURN_VERIFY_TOOLS` (a lab run counts as verification), `:4042` `_WORK_TOOLS`, `:7005`
  `_BATCH_WALL_TOOLS`, `:9882` `_MUTATING_TOOLS_FOR_DROP_NOTE`, the `:9698` job-id scrape (lab `job_id`s), and
  `:2454` `_EXTERNAL_EVIDENCE_TOOLS`.
- `core/evidence_gate.py:38` and `core/task_state.py:40` `EVIDENCE_TOOLS`.
- `skills/composed_skills.py:244` `MACRO_FORBIDDEN_STEP_TOOLS`: **add**, so a macro cannot create labs.
- If `lab` is advertised conditionally (hidden when unconfigured): `registry.py:1065`
  `CONDITIONALLY_ADVERTISED_BUILTIN_NAMES`, pinned by `tests/test_turnloop_r5_fixes.py`.
- `core/failure_replay.py:49` `READ_ONLY_TOOLS`: keep `lab` OUT (safe by omission).
- **At implementation start, re-grep every frozenset/tuple that lists `"execute"`** and decide `lab` for each. This list
  came from one reviewer pass, so treat it as a floor, not a ceiling.

**Add `lab` to:**
- `memory/egress.py:215` `OUTBOUND_TOOLS` with policy `"url_literals"`, like `execute`. A lab has Tor egress, so
  owner-private data in a URL inside a lab command must be scrubbed (§4LB).
- `core/isolation.py` replay/fork denylist (`:893` set). Replays, failure-replay and self-play must not create
  real labs. (Autopilot is NOT an isolated fork, so D7 still holds.)
- Members: `core/agent.py:325` `_MEMBER_ALLOWED_TOOLS` is an allowlist, so `lab` is **closed to members by
  default**. Keep it closed (labs are owner capacity with root + Tor). → Confirm with the operator if wanted.
- `core/metacog.py` tool→category map, `tool_failure.py` hints, `fallback_chains.py` (no fallback: a lab failure
  never falls back to running in the sandbox), `turn_state_check.py` (lab is NOT a local read).

**The old Postgres on the Mac** (`postgresql://ghost@127.0.0.1:5432/agent`, 43 MB, only demo tables + pgbench)
is used only by `postgres_admin` and `eval/behavioral.py`. Nothing else in `src/` uses `default_db`. Leave the
server running. Dropping the DB or uninstalling is the operator's call later.

**Docs** (memory `change-requires-tests-and-docs`, `docs-drift-is-the-rename`): remove/replace
`docs/tools/database.html`. Update `docs/tools.html`, `capabilities.html`, `installation.html`,
`cli_reference.html`, `tools/system.html`, `tools/fallback_chains.html`, `tools/execute.html`,
`tools/tool_failure.html`, `audit_fixes.html`, `tools/darkweb_search.html`, `tools/registry.html`,
`algorithms/dream_cycle.html` (13 doc files in all). New `docs/tools/labs.html` + a beginner page section, and a
`docs/labs/` README for the host. Also `PROJECT_JOURNAL.md` §4SX (there is no repo-level PROJECT_MAP/manifest; those
are per-project workspace files). Grep model-facing
prose for `postgres_admin` after removal (memory `workaround-outlives-its-bug`).

## 9. Implementation order (each phase: tests + docs, review by a reader who didn't write it)

**Phase A — host (`disorder`)**
1. `install.sh` (setup-time clearnet, D5): `daemon.json` (address pool `10.89.0.0/16`, log-opts) + Docker restart,
   host INPUT DROP for the pool + gateway-network hardening (persisted, e.g. `iptables-persistent` or a systemd oneshot), dirs,
   API key, the systemd unit with `After=tailscaled.service` + wait-for-tailnet-IP + `OOMScoreAdjust=-900`.
2. Images: base, tor-gateway, pg:18, patroni:18, web:1, mysql:8 (arm64, built locally).
3. Service: the contract API + the §3 deltas (origin/project/conversation labels, per-origin TTLs: ad-hoc 4 h/24 h,
   project 12 h/72 h (pending operator confirmation), owner pre-empts autopilot, autopilot may destroy another lab
   only if `last_owner_use_at` > 30 min ago, ad-hoc `members`, `http` probe, `spec`, `DELETE /v1/labs?project_id=`).
4. The transparent Tor gateway exactly as proven in §3b (fixed gw IP, `--dns`, route re-applied on Docker `start` events,
   v6 off, dns-search) + fail-closed checks at service start. Host-side forwarders owned by the service.
5. Integration tests on `disorder`: contract §11 + **every row of the §3b table as an automated test (reading data,
   not just connecting)** + pre-emption + reboot survival (route re-applied, volumes intact, jobs `lost`).

**Phase B — agent `lab` tool**
6. `tools/labs.py` (§5): `--labs-host`/`--labs-key-file` args, registry wiring (hidden when unconfigured), `EXIT CODE:`
   result shape, agent-side output cap, deadline clamp, egress scrub (`OUTBOUND_TOOLS` + `CODE_TOOLS`), isolation
   denylist + args clear, member closed, key redaction, transcripts, every tool list in §8 (re-grep first), prompt line.
7. Live probe with `X-Ghost-Origin`: "create a pg-single lab, select version(), destroy it".

**Phase C — projects**
8. Recipe files, `project_labs`, delete/archive hooks (outside the sync store method), `build_project_briefing`
   display, the `lab` task kind (+ ROUTE_LABEL, schema enum, advancer branch) + gate, the **autopilot lab route with the lab
   leaf allowlist**, put→exec claim binding, the `lab_capacity` HOLD that lifts automatically, and origin plumbing.

**Phase D — remove `postgres_admin`**
9. Delete the code, references, tests and docs (§8). Full suite once at the end (`scripts/run_suite.py -n 8` + lint).
10. Deploy: gate on `foreground_requests == 0`, `launchctl kill SIGTERM` (memory `operator-restarts-the-agent`).

**Acceptance (live, by the owner):** "create a self-hosted web site" → URL on the tailnet works from the owner's
browser; "destroy the site you just built" → gone, nothing labelled left; "create a 3 node patroni cluster",
"kill the leader and show failover", "destroy the whole patroni cluster"; "test this on MySQL 8"; a
project task closed DONE on a real lab transcript; an autopilot lab evicted by an owner create.

## 10. Operator answers (2026-11-03) and what is still open

1. Lock down `disorder`'s own outbound traffic after setup? **No.** Leave host egress open, log `build-images` (§3.13).
2. `lab` for Slack members? **No.** It stays closed (not in `_MEMBER_ALLOWED_TOOLS`).
3. TTLs: **yes.** 4 h idle / 24 h max for ad-hoc labs, and project labs get longer (the exact numbers need
   confirming: proposed **12 h idle / 72 h max** for labs with a `project_id`). The service needs `max_lifetime_s` per
   origin/project, so the global 24 h cap becomes a default.
4. The old local Postgres on the Mac: **it goes (operator 11-03).** Homebrew `postgresql@18`, data dir
   `/opt/homebrew/var/postgresql@18` (DBs `postgres` 19 MB + `agent` 43 MB, only demo/pgbench leftovers). Do it at the
   **end of phase D, after** the deploy without `postgres_admin`, because the live agent still holds a pooled connection
   to it until then. Steps: confirm nothing connects (`pg_stat_activity`), take a `pg_dumpall` to `~/Data/AI/backups/`
   (cheap insurance). It is a **system launchd daemon** `/Library/LaunchDaemons/com.local.postgres.plist` (RunAtLoad +
   KeepAlive) started through `~/Data/AI/bin/start-postgres.sh`, so `pg_ctl stop` would just be respawned. Instead:
   `sudo launchctl bootout system/com.local.postgres` → remove the plist → remove `start-postgres.sh` → `brew uninstall
   postgresql@18` → remove `/opt/homebrew/var/postgresql@18`. The launcher has no `GHOST_DEFAULT_DB` export (checked); only its
   header comment mentions `--default-db`.

## 11. Tomorrow's start checklist (2026-11-04)

**Read `docs/labs/UNIFIED_DESIGN.md` FIRST.** It is APPROVED (D1–D8, B1–B4) and its §8 phasing replaces phases B–D
here, **as revised by UNIFIED §12 (phasing) and §11 (the host is a runtime setting)**: A ∥ B0 → B → C (alongside services) → D (migrate
apps, then switch tools) → E. **Run an independent review (r2) of UNIFIED_DESIGN.md before B0 starts** (r1 reviewers hit the usage limit). B0 includes dropping the one live global binding (B4) at its deploy, not before.

1. Read `UNIFIED_DESIGN.md` (esp. §11–§14), then **`ghost-labs-contract-v2.md` = the Phase A spec** (it overrides
   `ghost-labs-manager-prompt.md` v1 where they differ), then this file's §3/§3b. **Superseded here:** §4 `exec detach`/§5's
   `job` action (→ `run`), §6.3 "archive destroys labs" (→ UNIFIED §3.4 + §13 #11 teardown), §6.7's lab-only leaf allowlist
   (→ `("file_system","execute","lab","run")`), §8 `skills/composed_skills.py` (→ `tools/composed_skills.py`), and every
   `100.64.0.13` (the image node; the agent is `eva` 100.64.0.12, a runtime setting).
2. `ssh disorder 'docker ps -a; docker network ls; sudo iptables -S INPUT'`: it should be clean (prototype removed
   11-03; only the `debian:trixie-slim` image + host `socat` remain).
3. **No open questions** (exposure X1–X4 answered 11-03, UNIFIED §15; day-1 item: the operator's Tailscale admin-console step for per-app names, then the test). Operator 11-03: autopilot creates **disposable machines only, when a task needs one** (UNIFIED §13 #21).
4. Phase A step 1. Do NOT touch `src/` until Phase A's integration tests pass on disorder.
5. Before Phase B: re-grep `src/ghost_agent` for every tool-name frozenset/tuple containing `"execute"` (§8).
6. Every phase: tests + docs + an independent review (memory `review-needs-a-reader-who-did-not-write-it`); the full suite
   once at the end (`scripts/run_suite.py -n 8` + lint); deploy gated on `foreground_requests == 0`.

## 12. Verification log

- **2026-11-03 r1:** independent read-only review (17 findings: 5 WRONG, 9 MISSING, 3 RISK). All folded in above:
  briefing site, `_EXTERNAL_EVIDENCE_TOOLS`, the experiment conflict narrowed (declaration not wording), the launchd Postgres,
  no repo PROJECT_MAP, the **autopilot lab route (critical)**, kind plumbing, the auto-lifting capacity hold, 10+ tool lists,
  args-not-env, key redaction, the `EXIT CODE:` shape, put→exec binding, the output cap, the out-of-repo launcher, the remaining
  references, and the sync delete hook.
- **2026-11-03 prototype on disorder:** §3b. Found and fixed the host-bridge hole and the DNS gap, and replaced
  the `-pub` network with host forwarders.

**11-04 Phase A built + live** (`interface/ghost-labs`, journal §4SX-j): all unit + integration suites pass on disorder incl. reboot.
Independent review r1 → r6 done (see below).

**11-04 Phase A verified** (journal §4SX-k): independent reviews r1–r6, every finding fixed and pinned (the last round
found one borderline MAJOR, now fixed). Unit 170; battery `scripts/mutate_4sx_labs.py` 114/114; integration on disorder
53/53 (egress, lifecycle, kept, agent-api, seed shell), then kept + agent-api + seed 30/30 again after the r6 fixes;
reboot pre/post PASS. Next: Phase B0.

**11-04 Phase B0 built, verified, LIVE** (journal §4SX-l): per-conversation focus (`conversation_focus` table, `RequestScope`
ContextVar), the global binding dropped at deploy (B4 logged), clients (CLI, web, Slack, ClockworkPi) send their session on
project and file calls. Reviews r1–r4: r1 7 findings, r2 2 MAJOR inside my r1 fixes, r3 MINOR only, r4 test gaps only.
Battery `scripts/mutate_4sx_b0.py` 76/76 (controls correct). Full suite 29,316 passed (the 2 failures it found were mine,
fixed and re-run green), lint OK. Live probe: switch / list / upload / bare download through agent and proxy. Next: Phase B.
- 2026-11-04 operator: **"i don't care about my current live apps. you can delete them."** → no migration of
  chess-coach (48e0373aaab3) / jiu-jitsu-calendar (9009ff89d8bb) in Phase D (UNIFIED D7 withdrawn). Their service rows
  and the 8-row registry go when the services module is removed (Phase D); the two projects + workspaces are deleted
  then too (backup first). With no app calling the agent API, nothing forces C and D into one deploy (UNIFIED §13 #6
  constraint gone); the D3 narrow path stays as built in Phase A but has no user until a new app needs it.

**11-04 Phase B built, verified, LIVE** (journal §4SX-m): the agent's `lab` tool (`tools/labs.py`, 13 actions) + its own
`lab_inventory.db`; configured by `GHOST_LABS_HOST` (launcher) + the 0600 key file, hidden when unconfigured; members and
replays refused; probes destroy only their own labs; workspace→lab-host reach refused in `execute`/`browser`; key
redacted; egress/provenance/strikes/evidence/outcome readers all know `lab`. Batch order: dependency waits in the
stoppable phase (`_batch_call_depends`). Reviews r1–r6 (r1 1 CRIT + 5 MAJOR, r2 7 MAJOR, r3 3 MAJOR inside my r2 fix,
r4 2 MAJOR, r5 3 MAJOR, r6 MINOR only). Battery `scripts/mutate_4sx_b.py` 122/122 (controls correct; the harness
no-op-control exit-5 bug fixed in `battery_cov._argfile_safe`). Full suite 29,490 passed + lint OK. Live probe:
create pg-single → exec → destroy (PostgreSQL 18.6), inventory/owner/origin right, host empty after. Next: Phase C.
**11-04 Phase C built, verified, LIVE behind a probe route** (journal §4SX-n): `run` (tools/runs.py) + `runs.db`
registry, workspace adapters over services/jobs (`start_detached`, `started_via`), lab detached jobs; NOT advertised to
the model (switch = Phase D); acceptance via `POST /api/operator/run-probe`. Reviews r1–r4, battery 101 mutants all
killed (1 equivalent dropped), suite 29,580 + lint OK. Live check found `run stop` missing a lab server (`timeout`'s own
process group) → agent `timeout --foreground` + labs HOST ends the job's whole group before `.rc` (deployed on disorder).
Gaps closed (§4SX-n-b): reconcile + boot line built (were missing), labs-host battery 121/121, labs-down + probe-wake
live-checked. Next: Phase D.
**11-04 Phase D2 built + verified locally** (journal §4SX-o): one project teardown (runs/services/jobs/labs) for tool +
console; reviews r1–r7 (last clean), battery 101 all killed; NOT deployed — operator: deploy once, after all phases.
**11-04 Phase D3 + D4 built + verified locally** (journal §4SX-p, §4SX-q): kept machines (keep/deploy/switch/rollback/
backup/restore, `app` runs, down notices, release on a kept app) and autopilot on disposable labs (leaf `lab`, lab
waits → PENDING + hold → NEEDS_USER after 3, leaf-tagged cleanup, project-owned schedules). Reviews r1–r6 (last two
MINOR only), agent + host batteries (survivors pinned). NOT deployed — one deploy after all phases.
**11-04 Phase D5 (the switch) built locally** (journal §4SX-r): `run` advertised; `manage_services`/`jobs` aliases;
every name set wired; promoted jobs = `job` runs; RUNNING briefing; leaf gets `run`; operator run-probe route removed.
**11-05 Phase D5 verified locally** (journal §4SX-r): reviews r1–r10. The last MAJOR was in r8; r9 and r10 found
MINOR only. The stored-row grading converged on ONE exit rule, `runs._headed` (`RUN ERROR` ⟺ failed,
`RUN UNRESOLVED` ⟺ no verdict). Battery: scripts/mutate_4sx_d5.py.
**11-05 Phase E built locally** (journal §4SX-s): `postgres_admin` and the local default DB removed; SQL goes to a
lab (`lab exec … stdin=`). The Mac's Postgres is removed AFTER the deploy. Next: full suite → ONE deploy (agent +
host kept.py/labs.py/service.py/api.py) → D1 (delete the two apps, drop the 6 legacy service rows, backed up) →
prod functional tests. **DONE 11-05 (journal §4SX-t): deployed, D1 done, Mac Postgres removed, prod 10/12 + tools 15/16 (all non-passes = test design / designed refusals).**
**11-05 end-to-end re-review r11–r16 (journal §4SX-u):** six fresh-eye rounds agent + host until clean; by-id jobs read ONE
registry record; lab paths resolve as file_system; host stats bounded, backfill removed. Deployed after battery + full suite.
