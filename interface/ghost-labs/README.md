# ghost-labs

A deterministic host service that runs disposable and kept lab containers for the ghost agent.
Spec: `docs/labs/ghost-labs-manager-prompt.md` (v1) + `docs/labs/ghost-labs-contract-v2.md` (v2; §11/§12 win).
Design: `docs/labs/UNIFIED_DESIGN.md`, `docs/labs/PLAN.md`.

No LLM, no decisions: the agent asks, the service does exactly that or returns an error with a hint.

## Install

On a Debian/Ubuntu host with Docker and Tailscale (any arch; images are built for `uname -m`):

```sh
rsync -a ghost-labs/ <host>:/opt/ghost-labs/        # as root
sudo /opt/ghost-labs/install.sh                     # idempotent, parameterless
```

It installs socat + a venv, writes `/etc/ghost-labs/{api_key,host_id,fw.env,config.yaml,templates/}` (templates/ = your own only), sets Docker's
`default-address-pools` to the lab pool and log rotation, installs `ghost-labs-fw.service` (before Docker),
`ghost-labs.service` and `ghost-labs-fwd@.service`, builds every image, starts the service and prints the
`GHOST_LABS_HOST` value. Setup is the only time the host uses the clearnet (image builds).

The agent side needs `GHOST_LABS_HOST=http://<tailnet IP>:8700` and the key from `/etc/ghost-labs/api_key`.

## Config (`/etc/ghost-labs/config.yaml`, every key optional; unknown keys refuse to start)

`bind_host: auto` (the Tailscale IPv4; never 0.0.0.0), `bind_port: 8700`, `ram_budget_mb: auto` (75 % of MemTotal − 1 GB),
`kept_mb: 1536`, `max_labs: 4` (disposable only), `reaper_interval_s: 60`, `published_port_range: [20000, 20999]`,
`exec_default_timeout_s: 120`, `exec_max_timeout_s: 1800`, `output_cap_bytes: 262144`, `file_max_bytes: 20971520`,
`address_pool: 10.89.0.0/16`, `gw_subnet: 10.88.0.0/24`, `api_subnet: 10.88.1.0/24`, `disk_floor_mb: 10240`,
`ttl: {adhoc: {idle_s: 14400, max_s: 86400}, project: {idle_s: 43200, max_s: 259200}, probe: {idle_s: 1800, max_s: 7200}}`,
`backup_keep: 7`, `crashloop: {max: 5, window_s: 600}`, `health_interval_s: 60`, `owner_grace_s: 1800`, `ts_services: true`.

## Security model

- Every API call needs `X-Labs-Key` (constant-time compare). `X-Labs-Origin: owner|autopilot|probe`; missing = autopilot.
  Autopilot can't touch kept machines (`owner_only`), can't destroy a lab the owner used in the last 30 min
  (`owner_active`), and never pre-empts. An owner create evicts the oldest autopilot disposable labs when the pool is full.
- Containers: never privileged, no Docker socket, no host bind mounts, only `NET_ADMIN`/`NET_RAW` may be added (NET_RAW
  is DROPPED unless the template lists it; unprivileged ping works through `ping_group_range`), pids 1024,
  memory = swap = `mem_mb`, IPv6 off, labelled `ghost.labs=1` + `ghost.lab.*`. Anything without `ghost.labs=1` is never touched.
- Egress `tor` (default) or `none`. Each lab is an `internal` network with its own /24 from `address_pool`. The shared
  gateway `ghost-labs-tor` sits at `.2` with TransPort/DNSPort and `FORWARD DROP`, `ip_forward=0`. Members get static IPs
  `.10–.127`, DNS = the gateway, and a default route via it (applied on create/start/rejoin, on Docker start/connect
  events and swept every reaper tick). `none` labs have no route and no external DNS.
- Host firewall (`ghost-labs-fw`, before Docker and tailscaled): INPUT DROP by source for the pool and both fixed subnets,
  AND by input interface (`glab-+`, `glabs-gw0`, `glabs-api0` — a member that forges its source is still dropped);
  `mangle FORWARD -i glab-+ ! -o glab-+ -j DROP` (a member never routes off its own bridge except via the gateway);
  `raw PREROUTING` DROP from the gateway and sidecar bridges toward 10/8, 172.16/12, 192.168/16, 100.64/10, 169.254/16.
  The service refuses to start, and refuses creates (`firewall_missing`), while any rule is missing.
- Published ports: a socat forwarder on `127.0.0.1:<port>` (+ the LAN IP only for `lan: true` kept apps) and
  `tailscale serve --https=<port>`, so apps are tailnet-only over HTTPS with a valid certificate.
- Secret values are never returned. exec/logs/files/http output replaces every stored secret (raw, base64, url, hex,
  JSON-escaped, across line breaks) with `<secret:NAME>`. This is transcript hygiene, **not** a security boundary.
- The agent-API sidecar (kept labs with `agent_api_url`) forwards only `GET|POST /api/game/*` to one tailnet IP:port,
  strips all but `Content-Type, Content-Length, Accept, X-Ghost-Service-Token`, follows no redirects, 64 KB / 1 MB caps.

## API (prefix `/v1`)

All of v1 §9 plus v2. Curl shape (key and origin on every call):

```sh
H=(-H "X-Labs-Key: $KEY" -H "X-Labs-Origin: owner" -H "Content-Type: application/json"); B=http://<host>:8700
curl "${H[@]}" $B/v1/health
curl "${H[@]}" $B/v1/templates
curl "${H[@]}" -X POST $B/v1/labs -d '{"template":"patroni-3","owner":"project:p1","label":"failover test"}'
curl "${H[@]}" -X POST $B/v1/labs -d '{"members":[{"name":"a","image":"ghost-lab/base:1","mem_mb":512}],"egress":"none"}'   # owner only (§4UL)
curl "${H[@]}" -X POST $B/v1/labs -d '{"template":"python-pg","overrides":{"mem_mb":{"app":1024},"ports":{"app":[8080,5000]}}}'  # owner only
curl "${H[@]}" $B/v1/labs                       # ?owner=project:p1
curl "${H[@]}" $B/v1/labs/lab_1234abcd
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/exec -d '{"member":"pg1","cmd":"psql -c \"select 1\"","timeout_s":60}'
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/exec -d '{"member":"pg1","cmd":"pgbench -T 600","detach":true}'   # 202
curl "${H[@]}" $B/v1/labs/lab_1234abcd/jobs     # and /jobs/<job_id>?tail_bytes=65536 ; DELETE kills it
#   a detached job is its process group: when the command ends, the rest of the group is ended before the exit code is written
curl "${H[@]}" -X PUT $B/v1/labs/lab_1234abcd/files -d '{"member":"pg1","path":"/root/x.sql","content_b64":"...","mode":"0644"}'
curl "${H[@]}" "$B/v1/labs/lab_1234abcd/files?member=pg1&path=/root/x.sql"
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/members/pg2/isolate   # stop|start|restart|kill|pause|unpause|isolate|rejoin
curl "${H[@]}" "$B/v1/labs/lab_1234abcd/logs?member=pg1&tail=200"
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/extend -d '{"ttl_idle_s":43200}'
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/http -d '{"member":"web1","port":80,"path":"/"}'
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/screenshot -d '{"member":"web1","port":80,"path":"/"}'
curl "${H[@]}" $B/v1/labs/lab_1234abcd/spec
curl "${H[@]}" -X DELETE $B/v1/labs/lab_1234abcd          # kept: ?include_kept=1 (&drop_data=1 after a final backup)
curl "${H[@]}" -X DELETE "$B/v1/labs?owner=project:p1"   # disposable labs of an owner (+include_kept=1, owner only)
curl "${H[@]}" -X POST $B/v1/gc                          # owner only: reap once + prune dangling images/build cache
# kept machines (owner origin only)
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/keep -d '{"app":{"name":"chess","member":"app","cmd":"python3 app.py","port":8080,"health":{"http":"/"}},"secrets":["APP_KEY"],"data":"shared","lan":false}'
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/deploy -d '{"archive_b64":"<tgz>","requirements":{"pip":["flask"],"build":"none"},"data_seed":["saves"]}'
curl "${H[@]}" -X PUT $B/v1/labs/lab_1234abcd/app/config -d '{"cmd":"gunicorn -b 0.0.0.0:8080 app:app","port":8080}'   # cmd|health|port|workdir|build|subdir
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/app/restart       # start|stop|restart (start leaves crashloop)
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/rehearse
curl "${H[@]}" -X POST $B/v1/labs/lab_v2/switch -d '{"from_lab":"lab_v1"}'
curl "${H[@]}" -X POST $B/v1/labs/lab_v2/rollback -d '{"restore_snapshot":false}'
curl "${H[@]}" $B/v1/labs/lab_1234abcd/backups ; curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/backups
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/restore -d '{"backup_id":"bk_..."}'
curl "${H[@]}" -X PUT $B/v1/labs/lab_1234abcd/secrets -d '{"secret_values":{"APP_KEY":"..."}}'
curl "${H[@]}" -X PUT $B/v1/previews -d '{"previews":[{"name":"chess dev","url":"https://...","project":"p1"}]}'
```

Errors: `{"error","message","hint",...}` with the codes in `ghost_labs/errors.py`. The Apps page is `https://<host DNS name>/`
(tailnet only, read-only: names, states, links).

## CLI

`ghost-labs status | list | templates | gc | url | build-images [name] | destroy <lab_id> [--include-kept] | destroy --all-disposable`.
`destroy --all` is an error on purpose.

## Templates

Shipped templates are read from the code's own `templates/` folder (`/opt/ghost-labs/templates/`), so syncing the
code deploys them (shipped: blank-dev, web, python-app, node-app, python-pg, mysql-single = tor; pg-single, patroni-3,
static-site = none).
`/etc/ghost-labs/templates/` holds only templates you add: a file there whose `name` is a shipped one is NOT loaded,
and the service logs a warning at start (rename it). A file there that does not load, for any reason, is also skipped with a
warning; it never stops the service (a shipped template that does not parse does). Two files there with one
name: the first (by file name) is loaded, the other is named in a warning. `install.sh` removes old identical copies of shipped ones there.
Keys: `name, description, egress, members[{name,image,mem_mb,cpus,ports,cap_add,env,ready}], env, readiness{member,cmd,check,timeout_s}, params,
app{member,port,cmd,health,build,service}` (§4UL: `app` fills what a keep leaves out; `service: true` = runs with nothing deployed).
`check` is `exit_code` or a predicate registered with `@ready_check` in `ghost_labs/labs.py`. Add a file, then
`systemctl restart ghost-labs`. Images must be `ghost-lab/*` and built (add `images/<name>/Dockerfile` + an entry in `cli.IMAGES`).

## Tests

`make test` (unit, Docker mocked) and `make itest` (real Docker, run as root on the labs host).
`AGENT_API_UPSTREAM=http://<tailnet ip>:<port>` enables the sidecar tests (an echo server on another tailnet node).
Reboot: `make itest-reboot-pre`, reboot, `make itest-reboot-post`.

## Kept versions, switch and recovery

- One version of an app (by **name**, tailnet-wide) is live = has the shared `/data` volumes mounted. A second `/keep`
  of the same app is **staged** (needs `data: shared`, the same `app.member`, and a live version that is itself
  `shared`); an app name used by another project is refused. keep / switch / rollback / delete-with-data of one app
  name are serialised (one lock per name, then the lab locks).
- `workdir` is replaced on every deploy: absolute, not `/`, `/data`, a system tree or an ancestor of `/opt/venv`.
- `data_seed`: relative, normalised, no path inside another, and **fixed per app**: the first seeding writes the
  list to `/data/.ghost-seeded`, which is authoritative afterwards; a staged version must declare the same list (or
  none) and `switch` refuses a different one. A seeded path is moved into `/data` only when absent there (existing
  data is never replaced); a symlink at the path, or a link parent that resolves outside the workdir, is refused
  before anything is created; a path in neither the archive nor `/data` is an error.
- switch / rollback snapshot **every member** first, then move the data. A failure (raised or a failed probe)
  reverses automatically; the switch needs the live version's deploy archive so it can. A rollback whose snapshot
  restore fails undoes that restore first; if even that fails, both versions stay stopped and **held** and the
  response lists the pre-restore snapshots. Once the data has moved, the switch is recorded before the port/name
  moves; a failure there is reported as `lease_error` (+ `last_error`) without undoing the switch.
- If the service dies mid-move, nothing is moved at the next start: both versions get a `last_error` saying so,
  and `switch` (to the version you want, `from_lab` = the other) works even when no version is live.
- Restore needs a mounted version, snapshots the restored member first (`pre_restore`) and streams from the file.
  A failure that did not touch `/data` puts the app back as it was (`touched: false`, `restarted` = whether it runs
  again). A failure after `/data` was touched leaves the app `stopped` and **held** (no restart, no keeper, no daily
  backup, not even after a reboot) with the step, the backup and the pre-restore snapshots in `last_error`. If the
  member's `/data` was cleared and not refilled, it carries the marker `/data/.ghost-restore-incomplete` (written
  before the clear, kept by it, removed only by a restore that completed): the pg image's `10-postgres.sh` then
  neither runs `initdb` nor starts postgres, even after a container restart, and `POST app/start` is refused
  (`not_ready`, `restore_incomplete`). Release: restore a good backup (it clears the marker), or
  `POST app/start?reinit=true` — removes the marker and starts postgres on what `/data` holds (an empty data dir is
  initialised — an EMPTY database only if the restore never got the files in); the protected snapshots stay protected
  for 7 days (`protected_until`, released by the first daily backup after it). A hold whose restore completed is
  released by a plain `app/start`. Why the app is held is in `hold_reason` (its own field, shown by the view; added to,
  never replaced by a later note).
- Backups: a failed copy leaves no file; the daily backup is retried after 1 h, not every tick. The final backup
  (delete with `drop_data`) covers every member, stops the app (also with a hook) and the lab goes in the same lock.
  A version with no data mounted (staged / switched out) never removes data volumes.
- Supervisor: a member container that stopped marks a running app `stopped` ("the app member's container is not
  running") and starts it when the container runs again; a container restart is not counted as an app crash.

## Deviations from the contract

- `/screenshot` helper has **no default route** (it reaches lab members only). Pages that load resources from the internet
  render without them. Stricter than §11.19, chosen so the page can't make the helper fetch anything.
- The rehearsal twin is created in the kept pool (`pool="kept"`, review 11-05): it never counts against, or evicts from,
  the disposable pool.
- Daily backups run from the app supervisor; the first one happens on the first tick after `/keep`.
- Export/import (§7, §11.17) is Phase E and not built yet.
- Daily backups cover the app member only; every member is snapshotted before switch / rehearse / delete-with-data.
- Data volumes of a retired (switched-out, `fresh`) version are not removed automatically.

## Review 11-05 (r11) — what changed on the host

- `reconcile` writes a lab's row BEFORE adopting its members (a restart mid-adopt lost kept rows); a lab caught in
  `creating` at restart is set `destroying` and reaped as `create_failed`; kept labels win over a twin's.
- Probe-made kept apps record what they claim (`probe_claims`: app, svc, port); retiring a probe app releases ONLY its
  own claims, never the owner's data volume, port or service name.
- Members are removed with their anonymous volumes (`remove(v=True)`); `gc` removes recorded ones a failed create left.
- Keep/switch re-check under the lab lock; one create lock covers the capacity check and the flip; eviction pre-checks
  `MemAvailable`; a create holds the lab lock through the build (no destroy mid-build).
- `jobs/{id}/kill` on a kept lab is owner-only like every kept mutation.
- Redaction works on bytes (non-UTF-8 files too), runs BEFORE any cut, covers the shlex-quoted form, and applies to the
  exec command echo and `actions.log`. An empty labs key refuses to start.
- Daily backups run on their own thread (a slow backup no longer stalls the app supervisor).
- Not migrated: networks made before r11 keep Docker's `br-*` bridge (`gc` lists them; recreate those labs), running
  containers keep NET_RAW until recreated.
- Probe kept apps made before r11 have no claims; none exist on the live host (0 kept labs at the r11 deploy) — if one ever appears, retire it by hand.
- r12: owner-create eviction counts what the autopilot victims really USE (`docker stats`: cgroup usage less
  file cache — since r13 active AND inactive; unreadable = 0), not their limits, and re-checks `MemAvailable` after each eviction: it stops
  as soon as the remaining victims can't make room (an idle 1000 MB lab no longer gets evicted for nothing).
- r12: a lab caught in `creating` at restart goes to the reaper only when it is disposable AND half-built (a spec
  member without its row/container, or a spec port without its lease); a kept lab, or a fully built one that died in
  its readiness wait, is `unready` (a kept lab is never reaped, so `destroying` stuck for ever; a kept lab r11 already
  left `destroying` is repaired to `unready` at the next start).
- r12: lock order is `[app lock] -> lab lock -> create_lock`. Eviction under `create_lock` only TRIES a victim's lock
  (all of them before anyone is evicted; a held one is skipped), so `/keep` holding a lab and waiting for
  `create_lock` no longer deadlocks an owner create into a 30 s `busy`; a create waits for Tor before taking
  `create_lock` (the capacity check and the row writes stay atomic under it).
- r12: app supervisor and switch texts for `actions.log` (crashloop, restart, interrupted move, reverse failed) are
  redacted BEFORE the 200-character cut (a secret straddling the cut was logged half).
- r13: eviction's "freed" estimate is cgroup usage less ALL file cache (`active_file` + `inactive_file`, v1
  `total_*`), clamped at 0 — `MemAvailable` already counts hot cache, so victims that only held cache are no longer
  evicted for a create that then fails anyway.
- r13: when only `MemAvailable` is short (budget and lab count fine), eviction skips victims that free ~0 MB and
  takes the largest users first; a budget/count shortfall still evicts the oldest first.
- r13: `/keep` refuses a lab that is not `ready` or `unready` (`not_ready` 409, `state` in the body): a
  `destroying`/`creating`/`failed` lab no longer becomes production.
- r13: the Tor gateway container is got-or-created under its own leaf `gateway_lock` (nothing is taken while it is
  held), and a docker 409 name conflict re-reads the gateway another process made — concurrent creates waiting for
  Tor outside `create_lock` no longer race into a 500.
- r14: eviction's `docker stats` calls run all at once under one overall deadline (`STATS_DEADLINE_S`, 5 s): they
  run under `create_lock` and the victims' lab locks, and one wedged call (docker's client timeout is 300 s)
  blocked every create. A container that does not answer in time, or errors, counts 0 MB freed.
- r15: a container whose earlier stats call is still running is not asked again (counts 0 MB, logged as such);
  each create used to start one more stuck thread per member.

## Review 11-11 (§4UD) — what changed on the host

- **The lab lock and a running command.** A foreground exec no longer keeps the lab's lock while its command
  runs (`labs.LabLock`: `to_exec` turns the hold into a "command running" mark). While a command runs: destroy
  (any caller, the reaper's max lifetime) and every member op go ahead, and the exec then answers `unknown_lab`;
  a second exec, a screenshot and every kept action wait and answer `busy` as before; the kept tick and an owner
  create's eviction skip the lab as before; the idle TTL does not fire (the exec touches the lab when it ends).
  The table is in the module docstring of `ghost_labs/labs.py` and pinned in `tests/unit/test_4ud_review.py`.
- **`POST /http` has one total deadline** (`HTTP_TOTAL_S`, 15 s: connect, headers and body). A watchdog shuts the
  socket at the deadline. What arrived is returned with `truncated: true` and `deadline_s`. `http.client`
  errors (a path it refuses, a bad status line) answer `status: null` with the reason, not 500. A refused
  connection is followed by one check inside the member (`/dev/tcp/127.0.0.1/<port>`): `loopback_only: true`
  when something listens there. `POST /screenshot` has one deadline for the probe, the browser and the copy
  (`SHOT_TOTAL_S`, 75 s).
- **Job state.** `.rc` present means `exited`, whatever the pid file says. `job_starts` stores the container's
  `StartedAt` per job: a job of an earlier start is `exited` (with `.rc`) or `lost`, and its pid is never
  signalled. `killed: true` only when this call sent the signal. The entrypoint removes
  `/var/log/lab/jobs/*.pid` at start and keeps `.log` and `.rc`. A job answer adds `output_truncated` and
  `log_path`.
- **`GET /logs` is bounded in bytes**: each part keeps its last `output_cap_bytes`; the container log is read
  as a stream with `follow=False`; the member cuts each file with `tail -c`. `truncated: true` when cut.
- **Kept labs.** `DELETE ?drop_data=1` keeps the data volumes while another kept version of the same app
  exists (decided under the app lock; `note`, `data_shared_with`), except a `data: fresh` version, which owns
  its volume. Logs, files, `/http`, screenshots and `GET /jobs/{id}` (its answer carries the output) of a kept
  lab are refused (`owner_only`) for a caller that may not touch it; its status stays readable. An exec re-reads
  the kept guard after it waited for the lab lock. A detached job whose lab is destroyed while it starts
  answers `unknown_lab` and leaves no job row. A detached start takes the lab lock for the launch only (over a
  running command) and checks the kept guard on the row read inside it. An exec whose exit code cannot be
  read while a state change holds the lock waits for it: a lab destroyed under the command answers
  `unknown_lab`, not 500.
- **TTL.** `ttl_idle_s` above the lab's maximum is clamped on create as on extend (`clamped_to_s`); below 1 or
  not an integer is `bad_request`. A lab summary adds `max_expires_at`.
- **`/v1/health` adds `features`** (`drop_data_guard`, `exec_free_lock`, `job_env`), so a client can tell this
  build from an older one.
- **Templates.** `pg-single` and `patroni-3` set `PGUSER=postgres`: a bare `psql` works. `python-app` says the
  app must listen on `0.0.0.0:8080`.
- **Compatibility with the agent deployed before this.** It sees: a create above the maximum succeed clamped
  (it refused); an exec whose lab was destroyed answer 404 (it answered after the command); `drop_data` leave the
  data when another version exists (the old agent asks for it only when its own list shows none); `run logs`
  of a kept app by autopilot answer 403; `killed` absent for a job that had already ended. New answer fields
  are ignored by it.

## Per-member readiness (§4UD F-A)

A template member may carry `ready: "<one command line>"`. `wait_ready` polls it with the template's own check
under the template's single `readiness.timeout_s`. The template check decides `state` (`ready`/`unready`) as
before; a member check never fails a create. The readiness record gains `members: {name: bool}`, and each
member in `GET /v1/labs/{id}` gains `ready` and `ready_cmd` when it declares a check. An older client ignores
both. The member checks run only after the template's check has passed (they never take a turn from it);
`ready` is `true`, `false` (the check ran and failed) or `null` (it never ran before the deadline; with a
template check that never passes, every member is `null`). Shipped: `pg_isready -q` (pg-single, patroni-3), `mysqladmin ping` (mysql-single).

## Cancel a foreground exec (§4UD F-B)

`POST /v1/labs/{id}/exec/cancel` with optional `{member, tag}`. `POST /exec` accepts an optional `tag`. The owner
ends any running command; an autopilot caller only commands that autopilot started with the same non-empty
tag, and none on a kept lab; a probe only a probe's. Nothing running: `200 {running: 0, cancelled: []}`.
Running but none the caller may end: `403 owner_only`. The exec that was cancelled answers `cancelled: true`,
`exit_code: null`, `killed: true|false`. The process group is killed by the command's pid file only when the
member's container `StartedAt` is the one the command began in, and `_kill_group` then looks (`kill -0`, up to
2 s): `killed` is true when the group was seen gone or the command's stream ended, false when the kill could not
be delivered (a paused member), the group is still there, there is no pid file yet (the command's shell has not
written it: nothing was signalled), or it was not sent and the stream is still open. The pid file is removed only
once the group is seen gone (or the command ended): a command not confirmed killed keeps it, so the retry below
reads it. A command's stdin (`stdin` in the body) goes to a 0600 file that the command's own shell opens and
deletes before anything else (before the command runs); the host also removes it after every exec, and when the
exec could not be created, because the shell may never have run (a missing workdir, a failed start). Limits: a
member paused in the milliseconds between the file's write and the shell's start refuses the host's removal too,
and the file stays until the member is removed; a pid file kept for a command that was not confirmed killed is
not removed when that command later ends by itself (small, harmless: ids are random). A timeout or cancel that
lands before the shell has started reads `killed: false`; the command then never runs (its stdin file is gone). A
timeout uses the same helper and field. While a command that could not be killed still runs, the lab keeps its
"command running" mark and the command stays in the cancel list: a later cancel sends the same kill again. Each `cancelled[]` item carries `outcome`:
`killed`, `not_killed`, `ended` (it ended by itself first) or `""` (no word within `CANCEL_CONFIRM_S`, 8 s).
An older client ignores both fields. `/v1/health` `features` names `exec_cancel` and `member_ready`.

## Folders (§4UD F-C)

`PUT /v1/labs/{id}/tree` `{member, path, archive_b64}` unpacks a tar(.gz) under `path`; `GET /v1/labs/{id}/tree
?member&path` returns a folder as `archive_b64` (tar.gz) with `entries`, `bytes`, `skipped`. Regular files and
directories only. A put with an absolute name, `..`, a backslash, a link, a device or a pipe is refused whole;
a get sends a hard link to a file already sent from the same folder as a copy of that file (Docker sends one
name of a hard-linked pair as the file and the other as a link to it), and those copies count while the answer is built: an answer
over `file_max_bytes` of content (what the agent accepts) is refused before more is built; it leaves symlinks, other hard links,
devices, pipes and names with a backslash out and names them (`skipped` holds the first 50, `skipped_total` the count). The get normalises `path` (`/data/.` is `/data`), refuses `/`,
and requires every archive entry's first component to be the folder's own name: anything else is an `internal`
error for the whole call, never an entry dropped or moved up a level. Both directions decompress under a byte
bound before `tarfile` reads (`tar_bytes_bounded`). Limits: `TREE_MAX_ENTRIES` (2000), the archive within
`file_max_bytes`, unpacked within four times that. The member receives an archive rebuilt from the checked
entries. Kept labs: put is the owner's (`guard_kept_mutation`), get is the owner's (`guard_kept_read`); file
contents are redacted as in `GET /files`. `GET /files` of a directory now adds `kind: "directory"` to its
`bad_request`. `/v1/health` `features` names `tree`.

## Known limitations

- A wedged `docker stats` call counts that victim as 0 MB, so an owner create can answer `budget_exceeded` where
  evicting it would have made room. A container whose earlier stats call is still running is not asked again
  (r15: each create used to start one more stuck thread per member).
- Redaction is hygiene, not a boundary (see above).
- When both the budget and memory are short, eviction is oldest-first for the budget and may evict two labs where one
  large one would have fixed both (extra churn, never a failure).
- A Tailscale Service name works only after the owner defines it in the admin console (`tcp:443`, tag `tag:labs-app`).
  Until it answers, `app.name_state` is `pending`/`unreachable` with a hint, and the port URL works.
- `apt`/`pip` deploy requirements install over Tor and can be slow; recreated members (rehearse/switch) reinstall them,
  so a switch can fail (and reverse) on a Tor hiccup.
- A crash during a rollback's snapshot restores is reported as an interrupted move; the pre-restore snapshot ids
  live only in the response; find them with `GET /backups` (kind `pre_restore`).
- Mutation battery: `python3 scripts/mutate_4sx_labs.py` (in the agent repo) — copies this tree, runs the unit suite per mutant.

## Templates, build step, in-place deploy (§4UL, 2026-11-12)

- New images `ghost-lab/static:1` (nginx on 8080 serving `/app`, no listing, dotfiles 404) and `ghost-lab/node-app:1`
  (Debian nodejs + npm), both FROM base. Build them on the host after syncing: `sudo ghost-labs build-images static`,
  `sudo ghost-labs build-images node-app`.
- New templates `static-site`, `node-app`, `python-pg`; `pg-single` is keepable (a watchdog service app).
- `POST /v1/labs` takes `overrides {mem_mb, ports, cap_add, cpus, egress}` (plain or `{member: value}`) on a template;
  `members` / `overrides` are owner-origin only.
- Deploy runs the app's build in `<workdir>.new` before the swap (`kept.AUTO_BUILD`: requirements.txt → `pip --target .deps`;
  package.json → `npm ci`/`npm install` + `npm run build --if-present`); `requirements.build` overrides per deploy, `"none"` = off.
  A failed build is `build_failed` (422) and the running version is untouched. The app starts with `kept.DEPS_PREFIX`.
- A deploy of an app that was `up` whose new version exits or answers ≥400 on its health path puts the previous version
  back: `<workdir>.prev` (the folder as it ran, no build; its `requirements.pip` reinstalled into `/opt/venv`), else its
  saved built folder, else its archive built again. Answers `rolled_back`, `failed_probe`, `data_note` (the old code runs on
  `/data` as the failed version left it; `pre_deploy_backup` names the backup from before, when the app had one)
  (+ `rollback_error`, `pip_note`). A live process not answering yet is `still_starting`. `built_not_saved`: the built
  folder was not saved (over `BUILT_MAX` or a docker error) — a switch/rollback/rehearse of it builds again.
- `PUT /v1/labs/{id}/app/config` changes cmd/health/port/workdir/build/subdir of a kept app (port moves the keyed lease;
  cmd/health/port/workdir restart it). Kept apps get `PORT` and `DATA_DIR=/data` in their env.
- Tests: `tests/unit/test_4ul_kept.py`.

### Review fixes (§4UL, 2026-11-12)

- The automatic revert reads `was_up` before the pre-deploy backup (a hook-less backup restarts the app) and puts
  `<workdir>.prev` back (the swap keeps the replaced folder) — no build. Each deploy saves the built folder as
  `<archive>.built.tar` (≤ `BUILT_MAX`); switch / reverse / rollback / rehearse / the twin unpack it instead of building
  again over Tor. Archives from before this have none and are built again once.
- Backups copy `/data` of every member; a `ghost-lab/pg:*` member has postgres stopped for its copy (`PG_STOP`/`PG_START`)
  — with a backup hook too (the hook runs first; re-review). A restore of a pg member stops postgres, restores the files,
  starts it and checks `pg_isready`; a failure names its `step`. Round 3: once a restore (single or set) has put back
  any pg member's files, the hook's `restore_cmd` is NOT run (its dump is older than the stopped copy) — `hook_note`
  says so; it runs only when no pg files were restored. `POST /restore` restores a whole set (same ts/kind), the app
  member last, and answers `pre_restore_by_member`; a failure holds the app only when `/data` was touched (an earlier
  member done, or past this member's snapshot/postgres stop — `touched` in the error) and records the member, the
  backup and the pre-restore snapshots in `last_error`; an untouched failure leaves the app as it was.
- Round 4: with ANY pg member in the lab, `restore_cmd` never runs (single, set, undo, rollback; `hook_note`); an
  `only_member` snapshot of a non-app member runs no hook; a failing hook never blocks a copy (`hook_error`, the file
  copy is taken); a restore opens its archive before the snapshot and `/data` clear, and the rotation never deletes
  its target/set or the ids in `protected_backups`; `held`/`watch_held` stop the keeper and daily backups until the app is
  started; a pending unserve is retried alone (after 1, 2, 4, 8, 16, 32 min; gives up after 6 failed retries), never by re-leasing; rollback
  undo goes db first.
- Round 5: a held database is held in the container too (the `/data` marker above; `10-postgres.sh` also refuses
  `initdb` during a restore — `GHOST_PG_RESTORING=1` — so a cluster-less archive fails instead of becoming an empty
  database; the restore installs the current script into the member first — round 6: mandatory and checked). `watch_held` is set with `held`, before
  `/data` is touched. Deploy on a held app installs without starting it (`held` in the answer says how to release);
  `rehearse` and `switch` refuse a held app (`not_ready`). A failed final backup (delete with `drop_data`) puts
  postgres, the keeper and the app back, sets `last_error` and the delete answers the failure — nothing is deleted.
  Protected snapshot ids live in `protected_backups` (not parsed from `last_error`); while held, a backup's note is
  appended to `last_error`. `hook_error` is passed through deploy/switch/restore/delete. A set restore opens and
  validates every member's archive before anything stops. A TCP app whose unserve retries gave up shows no stale
  name; `serve_failed` is not retried (the hint says so: use the port link). The untouched-restore error carries
  `restarted` (what happened), not `was_running`.
- Round 6: a pg restore first installs `images/pg/10-postgres.sh` into the member and checks it landed (sha256 of
  the file + executable) — a failure ends the restore UNTOUCHED (`step: install start script`, nothing stopped,
  marked or cleared). Then the keeper stops, `PG_QUIESCE` kills any running `start.d/10-postgres.sh` and its `initdb`
  (TERM, then KILL, bounded; still running = the restore fails untouched), postgres stops, `PG_QUIESCE` again, then
  the marker (a failed touch is untouched too: the marker is removed again and `restore_incomplete` put back) and the
  clear. `10-postgres.sh` checks the marker again right before `initdb` and before it starts postgres;
  `images/mysql/10-mysql.sh` refuses to start (or initialise) mysqld while the marker exists. An untouched failure
  puts every hold field back (`protected_backups` too). `restore_running` marks a restore under way. A successful
  member restore while another member is still `restore_incomplete` keeps the hold and its reason and does not wait
  for an app it cannot start. `rollback` of a held version needs `restore_snapshot: true`; `switch` checks the hold
  again inside its locks; a held app rotates no backups (any kind). Deploy on a held app keeps the build log
  (`probe.build_log`) and says there is NO automatic revert at the later start (deploy the previous version again).
- Round 7: a held app takes no pre-deploy copy (`pre_deploy_note`), and a restore retried while held reuses the
  pre-restore copy taken since the hold began (`hold_pre`, protected) instead of copying half-restored data. The clear
  runs `pgrep -x postgres && exit 3` in the same shell first: postgres back = the restore fails untouched (`step: check
  postgres is stopped`, the marker removed). The app is saved as soon as the marker is gone (`restore_incomplete`
  cleared; later errors are `step: finish`). The host's own `10-postgres.sh` must name the marker, else the restore
  fails untouched ("the host has an old 10-postgres.sh — redeploy images/").
- Known: `mysql`/`patroni` kept members are tarred live (only `ghost-lab/pg:*` is stopped), not crash-consistent.
- Round 3 also: an HTTP app's publish turns a TCP serve off; HTTP↔TCP unserve is decided from the live version too, a
  failed `unserve_service` keeps the name and is retried by `ensure_leases`; the keeper survives an owner stop and
  comes back on a member restart unless `watch_held` (a failed db restore); the -I note covers -E and keep/update_app.
- A saved built folder is unpacked whatever its single top folder is called (a workdir renamed since); another shape is
  built again from the archive (logged). The db keeper (`watch`) counts its pid only while it is still the keeper
  (`/proc/<pid>/environ`), the entrypoint clears `watch.pid`, and a watched member that restarts alone gets its keeper back.
- Unpacks run `chmod -R a+rX` (`tar --no-same-owner`). The build gets the app env (PORT, DATA_DIR, secrets). A cmd with
  `python -I` gets a deploy `note` (isolated mode ignores `.deps`).
- An app with no HTTP health path is published as raw TCP (`tailscale serve --tcp=<port> tcp://127.0.0.1:<port>`),
  answers `address` (host:port) instead of `url`, and gets no Service name; the restart reconcile keeps it TCP.
- Template `app` blocks may carry `wait_for` (host:port the app's start waits for, 90 s) and `watch` ([{member, cmd}]
  keepers in other members). python-pg uses both. Sizes: python-pg 256 + 512, pg-single 512, python-app 384 MB.
- `update_app`: a workdir move is checked (exists / nested) before anything stops; a failed port move puts the old lease
  back. The rehearsal twin copies member env/ready/ports.
- Deploy: sync, `sudo ghost-labs build-images base`, then `pg`, `static`, `node-app` (built on base; `mysql` too — its own base, round 6's marker check; the entrypoint
  clears `watch.pid`; the `pg` image carries the round-5 marker check — a running kept pg member keeps its old
  script until it is recreated or restored, since a restore installs the current one), `sudo systemctl restart
  ghost-labs`, then the agent. Tests: `tests/unit/test_4ul_review_kept.py`.
