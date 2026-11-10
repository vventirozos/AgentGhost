# Build `ghost-labs`: a disposable lab-container manager for a Raspberry Pi 5

You are building a small, deterministic service that runs on a dedicated Raspberry Pi 5
(arm64, Raspberry Pi OS Lite 64-bit / Debian trixie, Docker CE installed, NVMe boot,
Tailscale installed). An AI agent on another machine will call this service over HTTP to
create throw-away "labs" (groups of Docker containers), run commands inside them, break
them on purpose, and destroy them. Example requests the agent will serve:
"create a 3-node Patroni cluster and test failover", "set up a self-hosted web site",
"benchmark this SQL on PostgreSQL 18".

The service does the lifecycle work deterministically. It contains NO LLM logic and makes
no decisions for the agent. It must never leak resources and must never put the Pi itself at risk.

The HTTP API below is a **contract**: another program is written against it in parallel.
Do not rename fields, endpoints or error codes. If you must deviate, list every deviation
at the top of the README.

---

## 1. Stack and layout

- Python 3.11+, FastAPI + uvicorn, the `docker` Python SDK (docker-py), PyYAML, sqlite3
  (stdlib), pytest. No docker-compose at runtime: drive Docker through the SDK so every step
  is explicit and testable. Run blocking Docker calls in a thread pool and never block the event loop.
- Install root `/opt/ghost-labs` (code + venv), config `/etc/ghost-labs/`, state
  `/var/lib/ghost-labs/` (sqlite + logs), image recipes `/opt/ghost-labs/images/<name>/Dockerfile`,
  templates `/etc/ghost-labs/templates/*.yaml`.
- systemd unit `ghost-labs.service` (Restart=always, runs as root because it needs the Docker
  socket; nothing else on the Pi gets the socket).
- An idempotent `install.sh` that creates the venv, dirs, an API key (32 random bytes, hex,
  `/etc/ghost-labs/api_key`, mode 0600), installs the unit, builds the images, and enables and starts the service.
- A CLI `ghost-labs` (same codebase) with: `status`, `list`, `destroy <lab_id|--all>`,
  `gc` (run the reaper once and print what it removed), `build-images [name]`, `templates`.

## 2. Configuration (`/etc/ghost-labs/config.yaml`, all keys optional with these defaults)

```yaml
bind_host: auto          # auto = this machine's Tailscale IPv4 (tailscale ip -4); never 0.0.0.0
bind_port: 8700
ram_budget_mb: auto      # auto = 75% of MemTotal minus 1024 MB reserve for the OS
max_labs: 4
default_ttl_idle_s: 14400    # 4 h without any API call on the lab -> destroyed
max_lifetime_s: 86400        # 24 h hard cap regardless of use
reaper_interval_s: 60
published_port_range: [20000, 20999]   # host ports handed out to labs, bound on bind_host only
exec_default_timeout_s: 120
exec_max_timeout_s: 1800
output_cap_bytes: 262144     # per stream (stdout/stderr); keep the TAIL, report truncation
file_max_bytes: 20971520     # 20 MB per file transfer
tor_proxy: enabled           # see §6
```

If `bind_host: auto` cannot find a Tailscale IP, refuse to start with a clear log line. Never fall back to 0.0.0.0.

## 3. Security model (non-negotiable)

- Every request needs header `X-Labs-Key: <api_key>`, compared in constant time; otherwise `401`.
  `/v1/health` also requires it.
- Lab containers: **never** `privileged`, **never** mount the Docker socket, **never** bind-mount
  host paths (named volumes only, labelled per lab). Default Docker capabilities, `pids_limit: 1024`,
  `mem_limit` + `memswap_limit` equal to the member's `mem_mb`, `nano_cpus` from the template's `cpus`.
  Root inside the container is expected and fine: the agent has full access to its labs.
- A template may opt in to `cap_add: [NET_ADMIN]` per member (for tc/netem/iptables experiments).
  No other capability may be added.
- Every Docker object the service creates (containers, networks, volumes) carries the labels
  `ghost.labs=1`, `ghost.lab.id=<lab_id>`, `ghost.lab.member=<name>` (members only). The service must
  never touch an object without `ghost.labs=1`.
- Published ports bind on `bind_host` only, from `published_port_range`.

## 4. State, reconciliation and the reaper

- Docker labels are the source of truth. sqlite holds metadata only (template, params,
  created_at, last_used_at, expires_at, port leases, detached jobs).
- On startup: reconcile. Labs found in Docker but missing from sqlite are adopted, using their
  label timestamps, or are destroyed if those are unreadable. sqlite rows with no Docker objects are deleted.
  Labelled networks and volumes that belong to no live lab are removed. Log one line per action.
- Every API call that names a lab refreshes its `last_used_at`. `GET /v1/labs` (the listing) does not.
- The reaper runs every `reaper_interval_s` and destroys labs that are idle longer than their TTL, older than
  `max_lifetime_s`, or stuck in `creating` for more than 15 min. Destroy = stop + remove containers,
  networks, **volumes**, port leases, jobs. After a destroy, verify that nothing labelled with that lab id
  remains, and log loudly if something does.
- One asyncio lock per lab, so operations on the same lab never interleave. Creates are serialised globally
  so budget checks cannot race.

## 5. Resource budget

- Each template member declares `mem_mb`. A create succeeds only if
  `sum(mem_mb of live labs) + sum(mem_mb of the new lab) <= ram_budget_mb`, the live lab count
  is below `max_labs`, and `/proc/meminfo` MemAvailable ≥ the new lab's total + 512 MB.
  Otherwise fail with `budget_exceeded` (include needed, available and the live labs with their sizes).

## 6. Network and egress

- Each lab gets its own Docker bridge network `ghost-lab-<lab_id>` created with **`internal: true`**
  (no route to the internet or the LAN). Members resolve each other by member name through Docker DNS.
  Labs cannot see each other.
- Per-template `egress`:
  - `none` (default): internal network only.
  - `tor`: one shared container `ghost-labs-tor` (Debian + `tor`, SocksPort 0.0.0.0:9050,
    HTTPTunnelPort 0.0.0.0:8118), managed by the service and started on demand, is attached to
    the lab's internal network *and* to its own non-internal network. Lab members get
    `ALL_PROXY=socks5h://ghost-labs-tor:9050`, `HTTP_PROXY`/`HTTPS_PROXY=http://ghost-labs-tor:8118`
    (plus lowercase variants), and an apt proxy file `/etc/apt/apt.conf.d/99tor`. The lab has no other
    route out, so egress fails closed. Disconnect the Tor container from a lab's network when that lab
    is destroyed.
- `ports` in a template map a container port to a host port leased from `published_port_range`,
  bound on `bind_host`. The lab detail lists them as `{"container": 80, "host": 20003, "url": "http://<bind_host>:20003/"}`.
  Ports need a non-internal path. For members with ports, attach a second per-lab network
  `ghost-lab-<lab_id>-pub` that is **not** internal but has `com.docker.network.bridge.enable_ip_masquerade=false`
  and an iptables rule (installed by `install.sh`, persisted) that DROPs forwarded traffic from that
  bridge's subnet range to anything except replies. In short: inbound to published ports works, and outbound
  from the lab still fails. Write a test that proves outbound fails from a member with a published port.
  If you cannot make this airtight, implement `none`/`tor` only, make `ports` return
  `not_supported`, and say so in the README. Do not ship a leaky version.

## 7. Images (built locally on the Pi, arm64; the service never pulls at lab-create time)

- `ghost-lab/base:1`: debian:trixie-slim + tini, bash, sudo, ca-certificates, curl, wget, git,
  openssh-client, rsync, vim, nano, less, jq, htop, procps, psmisc, iproute2, iputils-ping, dnsutils,
  netcat-openbsd, tcpdump, strace, lsof, build-essential, pkg-config, python3, python3-venv,
  python3-pip, python3-dev, sqlite3, make, unzip, xz-utils. Locale C.UTF-8. Entrypoint: tini as PID 1 running
  `/usr/local/bin/lab-entrypoint` (runs `/etc/lab/start.d/*.sh` in the background, logging to
  `/var/log/lab/*.log`, then `exec sleep infinity`). Services therefore run as ordinary processes that the agent
  can stop and start freely without killing the container.
- `ghost-lab/pg:18`: FROM base + PostgreSQL 18 server, client and contrib from the PGDG apt repo (use
  `/usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y`), plus `pgbench`. A start script
  initialises a cluster on first start (in a named volume) and starts it. Listen on `*`, md5/scram for the lab
  network, local trust, superuser `postgres` password `lab`.
- `ghost-lab/patroni:18`: FROM pg:18 + `etcd-server`/`etcd-client` (Debian packages) + Patroni with the
  etcd3 extra in a venv on PATH (`patroni`, `patronictl`). The start scripts run etcd and then Patroni in the background.
- `ghost-lab/web:1`: FROM base + nginx + python3 venv tools; nginx serves `/srv/www` on port 80.
- Image tags are versioned. `build-images` is the only thing that pulls (base layers via normal Docker).
  Record each image's digest in `ghost-labs templates` output.

## 8. Templates (YAML, one file each; ship these four)

```yaml
# /etc/ghost-labs/templates/patroni-3.yaml
name: patroni-3
description: 3-node Patroni cluster (etcd colocated), PostgreSQL 18
egress: none
members:
  - {name: pg1, image: "ghost-lab/patroni:18", mem_mb: 512, cpus: 1.0}
  - {name: pg2, image: "ghost-lab/patroni:18", mem_mb: 512, cpus: 1.0}
  - {name: pg3, image: "ghost-lab/patroni:18", mem_mb: 512, cpus: 1.0}
env: {}            # template-wide env; members may override
readiness:
  member: pg1
  cmd: "patronictl -c /etc/patroni.yml list --format json"
  # ready when the JSON shows exactly one Leader and two members in state 'streaming'
  check: patroni_three_streaming
  timeout_s: 240
```

Also ship `blank-dev` (1 × base, 1024 MB, egress tor, readiness `true`), `pg-single`
(1 × pg:18, 768 MB, readiness `pg_isready -h localhost`) and `web` (1 × web, 512 MB, ports [80],
readiness `curl -fsS http://localhost/`). A readiness `check` is either the exit code of `cmd` (default)
or a named Python predicate over its stdout, registered in code (`patroni_three_streaming` is one).
Templates accept `params` (strings) that are substituted into env as `${param}` with declared defaults.
Unknown params give `bad_request`.

## 9. HTTP API (JSON in/out, prefix `/v1`, all responses have `Content-Type: application/json`)

Every lab-scoped response includes `"lab_id"`. Every error response is
`{"error": "<code>", "message": "<human sentence>", "hint": "<what to do next>", ...extra}` with these
codes and statuses:

| code | HTTP | extra fields |
|---|---|---|
| `unauthorized` | 401 | |
| `bad_request` | 400 | `field` |
| `unknown_template` | 404 | `templates` (list of names) |
| `unknown_lab` | 404 | `live_labs` (list of ids) — includes recently destroyed ids with reason, see below |
| `unknown_member` | 404 | `members` |
| `budget_exceeded` | 409 | `needed_mb`, `available_mb`, `live_labs` [{lab_id, template, mem_mb}] |
| `lab_limit` | 409 | `max_labs`, `live_labs` |
| `not_ready` | 409 | `state` |
| `busy` | 409 | `retry_after_s` (another op holds the lab lock > 30 s) |
| `not_supported` | 501 | |
| `internal` | 500 | |

For `unknown_lab`, keep a 24 h memory of destroyed labs. If the id is one of them, the message says
"lab <id> was destroyed at <iso time> (reason: idle_ttl|max_lifetime|deleted|create_failed)".

**Endpoints**

- `GET /v1/health` → `{"ok": true, "version": "…", "ram_budget_mb": N, "ram_used_mb": N,
  "mem_available_mb": N, "labs": N, "max_labs": N, "docker": "ok"|"<error>"}`
- `GET /v1/templates` → `{"templates": [{"name", "description", "members": [{"name","image","mem_mb"}],
  "total_mem_mb", "egress", "ports", "params": {"name": "default"}}]}`
- `POST /v1/labs` body `{"template": "patroni-3", "label": "optional free text ≤80 chars",
  "ttl_idle_s": optional (≤ max_lifetime_s), "params": {}, "wait": true}`
  → `201` lab detail. With `wait: true` (default) it returns only after readiness passes or times out.
  On timeout: return `201` with `"state": "unready"` and the last readiness output, and **do not** destroy the lab,
  so the agent can debug it. If creation itself fails partway, roll back everything and return the error,
  noting the rollback in `message`.
- `GET /v1/labs` → `{"labs": [<lab summary>]}`. Summary = `lab_id, template, label, state, created_at,
  last_used_at, expires_at, total_mem_mb`.
- `GET /v1/labs/{lab_id}` → lab detail = summary + `members: [{"name", "container_id", "status"
  (running|exited|paused), "ip", "image", "mem_mb", "ports": [...], "isolated": bool}]`, `network`, `egress`,
  `readiness: {"passed": bool, "output": "…tail…", "checked_at"}`.
- `POST /v1/labs/{lab_id}/exec` body `{"member": "pg1", "cmd": "string run via bash -lc",
  "timeout_s": 120, "workdir": "/root", "env": {}, "stdin": "optional string", "detach": false}`
  → `{"lab_id","member","cmd","exit_code","stdout","stderr","stdout_truncated","stderr_truncated",
  "duration_s","timed_out"}`. On timeout, kill the exec's process group inside the container and return
  `timed_out: true, exit_code: null` (HTTP 200, not an error). With `detach: true` → `202
  {"lab_id","member","job_id","cmd"}`. The process runs with `setsid`, writes its own pid, and output goes to
  `/var/log/lab/jobs/<job_id>.log` inside the container.
- `GET /v1/labs/{lab_id}/jobs/{job_id}?tail_bytes=65536` → `{"job_id","member","cmd","state"
  (running|exited|lost),"exit_code","output_tail","started_at","ended_at"}`.
  `DELETE` on the same path → kills the job's process group → the same shape.
- `PUT /v1/labs/{lab_id}/files` body `{"member","path","content_b64","mode": "0644"}` →
  `{"lab_id","member","path","bytes"}`. Parent dirs are created.
  `GET /v1/labs/{lab_id}/files?member=&path=` → `{"lab_id","member","path","bytes","content_b64"}`.
  Both obey `file_max_bytes` (`bad_request` with the size if over). Use put_archive/get_archive.
- `POST /v1/labs/{lab_id}/members/{member}/{action}` with action ∈ `stop|start|restart|kill|pause|unpause|isolate|rejoin`
  → `{"lab_id","member","action","status","isolated"}`. `isolate` disconnects the member from the lab
  network (a network partition). `rejoin` reconnects it with the same IP and aliases.
  These are the chaos primitives, so make them exact and idempotent.
- `GET /v1/labs/{lab_id}/logs?member=&tail=200` → `{"lab_id","member","logs"}` (container stdout/stderr
  plus `/var/log/lab/*.log` tails, each section headed by its file name).
- `POST /v1/labs/{lab_id}/extend` body `{"ttl_idle_s": N}` → summary (still capped by `max_lifetime_s`).
- `DELETE /v1/labs/{lab_id}` → `{"lab_id","destroyed": true,"removed": {"containers": N,"networks": N,
  "volumes": N}}`. Idempotent: destroying a recently destroyed lab returns the same shape with zeros.

Lab ids: `lab_` + 8 lowercase hex characters. State values: `creating | ready | unready | destroying`.

## 10. Logging

One INFO line per state-changing action to the journal and `/var/lib/ghost-labs/actions.log`
(JSON lines: ts, lab_id, action, member, outcome, duration_ms). Exec lines record the command
truncated to 200 chars, never its output. Never log the API key.

## 11. Tests (required, and they must pass on the Pi)

- Unit (pytest, Docker mocked): auth, budget arithmetic and races, template parsing/params,
  error shapes for every code, output truncation keeps the tail, reaper decisions (idle / lifetime /
  stuck creating), reconciliation (adopt, orphan cleanup), destroyed-lab memory.
- Integration (`tests/integration/`, real Docker, marked so they can be skipped):
  1. `pg-single`: create → `psql -c 'select version()'` shows 18 → destroy → nothing labelled remains.
  2. `patroni-3`: create (wait) → find the leader → `kill` it → within 90 s another member is the leader →
     `start` the old one → it rejoins as a replica → destroy → no leftovers.
  3. Partition: `isolate` pg2 → from pg1, `ping -c1 -W1 pg2` fails → `rejoin` → it succeeds.
  4. Egress: in a `none` lab and in a `web` lab, `curl -m5 https://example.com` fails. In a `tor` lab it succeeds
     through the proxy, and `curl --noproxy '*' -m5 https://example.com` fails.
  5. Reaper: create with `ttl_idle_s: 5`, wait, run `ghost-labs gc` → lab gone, volumes gone.
  6. Budget: creating more than the budget allows returns `budget_exceeded` with the right numbers.
- Provide `make test` and `make itest`.

## 12. Deliverables

The code, `install.sh`, the systemd unit, the Dockerfiles, the four templates, tests, and a README
covering install, config, the API (copy §9 exactly, with a curl example per endpoint), the
security model, how to add a template, and a "Known limitations" section. Do not stop at a
plan: produce the full, working implementation, run the tests, and report the actual
results. Report failing tests as failing.

**Out of scope, do not build:** a web UI, multi-host support, user accounts, an LLM, Docker
socket access for labs, auto-updates, anything listening outside the Tailscale IP.
