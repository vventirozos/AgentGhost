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

It installs socat + a venv, writes `/etc/ghost-labs/{api_key,host_id,fw.env,config.yaml,templates/}`, sets Docker's
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
- Containers: never privileged, no Docker socket, no host bind mounts, only `NET_ADMIN` may be added, pids 1024,
  memory = swap = `mem_mb`, IPv6 off, labelled `ghost.labs=1` + `ghost.lab.*`. Anything without `ghost.labs=1` is never touched.
- Egress `tor` (default) or `none`. Each lab is an `internal` network with its own /24 from `address_pool`. The shared
  gateway `ghost-labs-tor` sits at `.2` with TransPort/DNSPort and `FORWARD DROP`, `ip_forward=0`. Members get static IPs
  `.10–.127`, DNS = the gateway, and a default route via it (applied on create/start/rejoin, on Docker start/connect
  events and swept every reaper tick). `none` labs have no route and no external DNS.
- Host firewall (`ghost-labs-fw`, before Docker and tailscaled): INPUT DROP for the pool and both fixed subnets;
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
curl "${H[@]}" -X POST $B/v1/labs -d '{"members":[{"name":"a","image":"ghost-lab/base:1","mem_mb":512}],"egress":"none"}'
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
curl "${H[@]}" -X POST $B/v1/labs/lab_1234abcd/deploy -d '{"archive_b64":"<tgz>","requirements":{"pip":["flask"]},"data_seed":["saves"]}'
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

YAML in `/etc/ghost-labs/templates/` (shipped: blank-dev, web, python-app, mysql-single = tor; pg-single, patroni-3 = none).
Keys: `name, description, egress, members[{name,image,mem_mb,cpus,ports,cap_add,env}], env, readiness{member,cmd,check,timeout_s}, params`.
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
- Restore needs a mounted version, snapshots the restored member first (`pre_restore`) and streams from the file;
  on failure the app is `stopped` and **held** (no restart, not even after a reboot) with the snapshot id in
  `last_error`; `POST app/start` releases the hold.
- Backups: a failed copy leaves no file; the daily backup is retried after 1 h, not every tick. The final backup
  (delete with `drop_data`) covers every member, stops the app (also with a hook) and the lab goes in the same lock.
  A version with no data mounted (staged / switched out) never removes data volumes.
- Supervisor: a member container that stopped marks a running app `stopped` ("the app member's container is not
  running") and starts it when the container runs again; a container restart is not counted as an app crash.

## Deviations from the contract

- `/screenshot` helper has **no default route** (it reaches lab members only). Pages that load resources from the internet
  render without them. Stricter than §11.19, chosen so the page can't make the helper fetch anything.
- The rehearsal twin is checked against the kept pool, but while it lives it is stored as a disposable lab (it shows in
  `disposable_mb_used` for its few minutes).
- Daily backups run from the app supervisor; the first one happens on the first tick after `/keep`.
- Export/import (§7, §11.17) is Phase E and not built yet.
- Daily backups cover the app member only; every member is snapshotted before switch / rehearse / delete-with-data.
- Data volumes of a retired (switched-out, `fresh`) version are not removed automatically.

## Known limitations

- Redaction is hygiene, not a boundary (see above).
- A Tailscale Service name works only after the owner defines it in the admin console (`tcp:443`, tag `tag:labs-app`).
  Until it answers, `app.name_state` is `pending`/`unreachable` with a hint, and the port URL works.
- `apt`/`pip` deploy requirements install over Tor and can be slow; recreated members (rehearse/switch) reinstall them,
  so a switch can fail (and reverse) on a Tor hiccup.
- A crash during a rollback's snapshot restores is reported as an interrupted move; the pre-restore snapshot ids
  live only in the response; find them with `GET /backups` (kind `pre_restore`).
- Mutation battery: `python3 scripts/mutate_4sx_labs.py` (in the agent repo) — copies this tree, runs the unit suite per mutant.
