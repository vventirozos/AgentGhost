# ghost-labs contract v2: delta over `ghost-labs-manager-prompt.md` (v1)

Status: **the Phase A spec, 2026-11-03; independently reviewed (r3), all 23 findings folded into §11, which overrides §1–§10.** Where v2 and v1 differ, **v2 wins**. Sources: PLAN.md §3/§3b, UNIFIED_DESIGN.md §6/§11/§12/§13.
Anything v1 says that is not changed here still holds (auth header, error shape, lab ids, the per-lab lock, the destroyed-lab memory,
logging, output caps, file transfer).

## 1. Host, install, config (UNIFIED §11)
- Nothing host-specific in code. `install.sh` is parameterless and idempotent on Debian/Ubuntu with Docker + Tailscale, on **any
  arch** (images built for `uname -m`). At the end it prints the `GHOST_LABS_HOST` value + the key path.
- `install.sh` also:
  - writes `/etc/docker/daemon.json`: `default-address-pools: [{base: <pool>, size: 24}]` (config `address_pool`,
    default `10.89.0.0/16`), plus `log-opts max-size 10m, max-file 3`, then restarts Docker;
  - installs the **host INPUT DROP** for the pool (NEW from the pool → host), plus the gateway-network hardening (DOCKER-USER DROP
    from the gateway net to RFC1918/100.64/10/169.254/16, INPUT DROP from it), persisted by a systemd oneshot `ghost-labs-fw.service`;
  - adds a drop-in so `ghost-labs.service` runs `After=tailscaled.service docker.service` and waits for the tailnet IP, with
    `OOMScoreAdjust=-900`;
  - generates `host_id` (uuid4) in `/etc/ghost-labs/host_id`.
- The service **refuses to start** if the firewall rules are missing or the tailnet IP is absent (fail closed).
- New config keys (defaults): `address_pool: 10.89.0.0/16`, `kept_mb: 1536`, `disk_floor_mb: 10240`,
  `ttl: {adhoc: {idle_s: 14400, max_s: 86400}, project: {idle_s: 43200, max_s: 259200}, probe: {idle_s: 1800, max_s: 7200}}`,
  `backup_keep: 7`, `crashloop: {max: 5, window_s: 600}`, `health_interval_s: 60`, `owner_grace_s: 1800`.

## 2. Egress (replaces v1 §6; proven in PLAN §3b)
- `egress: tor` (**default**) | `none`. There is no other mode.
- One shared gateway container `ghost-labs-tor` (image `ghost-lab/tor-gw:1`: tor + iptables, as in §3b). It gets a **fixed IP `.2`** on every
  lab network, `TransPort 0.0.0.0:9040 IsolateClientAddr`, `DNSPort 5353`, `ClientRejectInternalAddresses 1`, the data dir on a named
  volume, `nat PREROUTING -i <lab-if> tcp→9040, udp 53→5353`, and `FORWARD DROP`. A `tor` create waits for `Bootstrapped 100%`, else
  `tor_unavailable` (503).
- Members are created with `dns=[<gw .2>]`, `dns_search=["lab.invalid"]` and `sysctls net.ipv6.conf.all.disable_ipv6=1`. The default route
  goes via the gateway, applied with `nsenter -t <pid> -n ip route replace default via <gw>` **on every Docker `start` event** (an events
  listener) **and at service start for every running member** (reconcile), before serving.
- `egress: none`: no gateway on the network, no route.
- The v1 `-pub` network is **not built**. Published ports go through host forwarders (§4).

## 3. Lab model changes
- Create body adds: `lifetime: disposable` (only; `kept` is reached via `/keep`), `origin: owner|autopilot|probe`,
  `owner: "project:<id>" | "conversation:<key>"`, `lineage` (root project id; required for kept), `purpose` (≤120),
  `members: [...]` as an alternative to `template` (catalog images only), `agent_api_url` (optional; see §6).
- Labels add `ghost.lab.owner`, `ghost.lab.origin`, `ghost.lab.lineage`, `ghost.lab.lifetime`. The summary returns them + `ram_mb`.
- TTL per owner class (§1 `ttl`). **Kept labs have no TTL.** They are exempt from the reaper, from reconcile-destroy (an unreadable
  kept lab is **adopted and flagged**, never destroyed) and from `destroy --all` (which becomes `--all-disposable`).
- Budget: `ram_budget_mb` = kept pool (`kept_mb`) + disposable pool (the rest). `max_labs` counts disposable only. A kept check
  includes the version-switch overlap. `budget_exceeded` adds `pool: kept|disposable`.
- **Pre-emption:** an `origin=owner` create that doesn't fit the disposable pool destroys the oldest `origin=autopilot` disposable
  lab(s) and returns `evicted: [{lab_id, owner}]`. Autopilot never pre-empts.
- **Destroy guard:** `origin=autopilot` may destroy a lab whose `last_owner_use_at` is older than `owner_grace_s`, and never a kept
  lab. `409 owner_active` otherwise.
- `DELETE /v1/labs?owner=<owner>` destroys all disposable labs of an owner. Kept labs need `?include_kept=1` (project delete only).
- **Disk:** a create is refused with `disk_low` (507) under `disk_floor_mb`. `gc` also prunes dangling images/build cache.

## 4. Ports
- Forwarders: one systemd unit per port, `ghost-labs-fwd@<port>.service` (socat `TCP-LISTEN:<port>,bind=<bind_host>,fork →
  <member ip>:<cport>`). They are reconciled at service start. Ports come from `published_port_range`.
- **Lease keys:** disposable = per lab. **Kept app = per `(lineage, app name)`**, stable across recreate, rehearsal and versions.

## 5. Kept machines and apps (new)
- `POST /v1/labs/{id}/keep {app: {name, cmd, port, health: {http: "/"|null, tcp: true}, workdir: "/app"}, data: "shared"|"fresh",
  secrets: [NAME...], secret_values: {NAME: value} (agent-supplied, e.g. GHOST_SERVICE_TOKEN; stored like generated ones, never returned), cron: ["…"], backup: "<cmd>"|null}` turns a disposable lab into a kept one (owner must be a project, else
  `bad_request field=owner`).
  - It creates the named volume `ghost-labdata-<lineage>-<app>` mounted at `/data` (or attaches the existing one for `shared`).
  - It generates the missing secrets (32 bytes, base64) and stores them in sqlite (0600). They are injected as env on every start, and
    **values are never returned** (names only).
- **App supervision:** the manager starts `cmd` (`setsid`, pid recorded). The health probe runs every `health_interval_s`; on failure or exit
  it restarts with backoff (5 s, 15 s, 45 s…). `crashloop` (state) after `crashloop.max` failures in `window_s` → stopped and reported.
  At service start and after host reboot, every kept lab's app is started.
- `POST /v1/labs/{id}/deploy {archive_b64 | archive_path, requirements: {pip: [...], apt: [...]}, data_seed: [paths]}`:
  - unpacks to `/app` (replacing it) and installs deps over Tor;
  - on the **first** deploy only, moves `data_seed` paths from the archive into `/data` (refused afterwards: `data_exists`);
  - restarts the app and returns the probe result.
- `POST /v1/labs/{id}/rehearse`: snapshot `/data` → recreate the containers from the spec **keeping the data volume** → redeploy the last
  archive → probe. Also creates a **disposable twin** from the same spec + archive with an empty `/data`, probes it, destroys it.
  Returns both results.
- `POST /v1/labs/{id}/switch {from_lab}` (versions): snapshot → stop `from_lab`'s app → attach the shared data volume (`data: shared`) →
  start → probe → move the port lease. `from_lab` is kept **stopped** for 24 h as the rollback target, then destroyed.
  `POST /v1/labs/{id}/rollback {restore_snapshot: false}` reverses it.
- **Backups:** daily, and before deploy/switch/delete. Run `backup` inside the member if declared (e.g. `pg_dump > /data/backup.sql`), else
  stop-app → tar the volume → start-app. Keep `backup_keep` under `/var/lib/ghost-labs/backups/<lineage>/<app>/`.
  `GET …/backups`, `POST …/restore {backup_id}`.
- `cron` lines are installed in the member's crontab (the base image includes `cron`, started by `start.d`).

## 6. The agent-API path (D3, r2 #1/#5)
- On a kept lab whose create/keep carried `agent_api_url`, the manager runs a sidecar `ghost-labs-agentapi-<lab>` on the lab network,
  reachable from members as `agent-api:8000`, which members get as `GHOST_API`. The sidecar is an HTTP reverse proxy (tiny, e.g. Caddy or a
  50-line aiohttp) whose upstream is `agent_api_url`. It:
  - allows only `GET|POST /api/game/*`;
  - passes only the `X-Ghost-Service-Token` header and strips `X-Ghost-Key`, `X-Ghost-Requester`, `X-Ghost-Origin`, `Cookie`, `Authorization`;
  - caps the body at 64 KB;
  - is only attached to that lab's network + one egress network allowed (DOCKER-USER) to reach **only** `agent_api_url`'s host:port.
- The manager validates `agent_api_url` as a tailnet (100.64/10) host.
- The service token is minted by the **agent** (its run registry) and passed in `secrets` as `GHOST_SERVICE_TOKEN`.

## 7. Other new endpoints
- `POST /v1/labs/{id}/http {member, port, path, method: GET}` → `{status, headers, body_head (16 KB)}`, fetched from the host over the lab
  network.
- `POST /v1/labs/{id}/screenshot {url, width: 1280, height: 800}` runs a helper container `ghost-lab/browser:1` (headless Chromium) on the lab
  network and returns `{png_b64}` (≤ 2 MB).
- `GET /v1/labs/{id}/spec` → the exact create + keep + deploy recipe (no secret values).
- `POST /v1/labs/{id}/export` → a tarball path on the host (spec, archive, data volume, secrets **encrypted with the API key**, backups);
  `POST /v1/import {path}`.
- **Redaction:** exec/logs/files/http output has every stored secret value replaced with `<secret:NAME>` before returning.
- `GET /v1/health` adds `host_id`, `disk_free_mb`, `kept_mb_used`, `disposable_mb_used`, `tor: "ok"|"bootstrapping"|"down"`, and
  `apps: [{lab_id, app, state: up|restarting|crashloop|stopped}]`.

## 8. Error codes added
`tor_unavailable` 503, `disk_low` 507, `owner_active` 409, `data_exists` 409, `crashloop` (a state, not an error),
`not_kept` 409 (a kept-only action on a disposable lab), `owner_only` 403 (`origin=autopilot` on `/keep`, `/deploy`, `/switch`, `/rehearse`, `/rollback`, `/restore`), `labs_unreachable` (**agent-side only**, never returned by the service).

## 9. Images (adds to v1 §7)
`ghost-lab/tor-gw:1`, `ghost-lab/python-app:1` (base + flask, gunicorn, requests, common libs), `ghost-lab/browser:1` (base +
Chromium + a playwright screenshot script), `ghost-lab/mysql:8`. The base adds `cron`. All are built at install/`build-images`
(the only clearnet use, D5).

## 10. Integration tests added (Phase A acceptance, UNIFIED §14)
Every PLAN §3b row as a data-reading test; the host INPUT DROP (a member gets no banner from host sshd); agent-api proxy: `/api/game/games` ok,
`/api/chat` 403, the `X-Ghost-Key` header stripped, another tailnet host unreachable; kept lab survives `gc`, the reaper and
`destroy --all-disposable`; app crash → restarted; 5 crashes → `crashloop`; host reboot → routes, forwarders and apps back; deploy →
redeploy keeps `/data`; rehearse keeps data and the twin passes; switch + rollback; a backup with a `pg_dump` hook restores; secret
redaction in exec output; `disk_low` and `owner_active` paths; pre-emption evicts only autopilot disposable labs; `origin=autopilot` gets `owner_only` on every kept-machine action.

---

## 11. Review r3 (independent, 2026-11-03): binding fixes. Where this section and §1–§10 differ, **§11 wins**

Verified on the labs host before folding: Docker refuses `--ip` on a network whose subnet came from `default-address-pools`, and
accepts it on an explicit `--subnet … --ip-range …` network. The FORWARD chain order is `ts-forward` → `DOCKER-USER` → `DOCKER-FORWARD`
(tailscaled's chain comes first).

### HIGH
1. **Network IPAM.** The manager allocates each lab network's /24 itself from `address_pool` (tracked in sqlite) and creates it with
   `IPAMPool(subnet=<s>/24, gateway=.1, iprange=.128/25)`. It connects `ghost-labs-tor` at **`.2` before any member starts**. Members get
   **static IPs from `.10–.127`**, stored in sqlite and reused on recreate/rejoin. (Docker's `default-address-pools` stays only as a
   fence for non-lab networks.)
2. **Forwarders target static member IPs.** Each `ghost-labs-fwd@<port>` reads `TARGET=<ip>:<cport>` and `BIND=<ip>` from
   `/etc/ghost-labs/fwd/<port>.env`. A lease move rewrites the file and restarts the unit. Units have `Restart=always`,
   `After=tailscaled.service`, and socat runs with `max-children=64`.
3. **Data volumes are not lab objects.** Labels `ghost.labs=1, ghost.data.lineage, ghost.data.app, ghost.data.member`, and **no
   `ghost.lab.id`**. Destroy, the reaper, reconcile and `gc` never remove them. Only `DELETE /v1/labs/{id}?include_kept=1&drop_data=1`
   (origin owner) removes one, after a final backup.
4. **Gateway hardening survives restarts.** `ghost-labs-tor` runs with `net.ipv4.ip_forward=0` and `restart: unless-stopped`. Its
   entrypoint installs `nat PREROUTING ! -i <egress-if>` (tcp → 9040, udp/53 → 5353) and `filter FORWARD DROP` **before** starting tor,
   and the rules key on the egress interface found by its subnet, not on lab interface names. Destroy **disconnects the gateway,
   sidecar and helper containers** before removing a lab network.
5. **Static firewall, ordered before Docker and tailscaled.** `install.sh` creates two fixed networks:
   - `ghost-labs-gw` (bridge `glabs-gw0`, fixed subnet), the gateway's egress;
   - `ghost-labs-api` (bridge `glabs-api0`), the sidecars' egress.

   A oneshot `ghost-labs-fw.service` (`Before=docker.service`, `WantedBy=multi-user.target`) installs
   **`raw PREROUTING -i glabs-gw0|glabs-api0 -d 10/8,172.16/12,192.168/16,100.64/10,169.254/16 -j DROP`** plus the host INPUT DROP for
   the pool and both fixed subnets. The raw table runs before `ts-forward`/DOCKER-USER, so tailscaled can't bypass it. The manager inserts
   the per-agent ACCEPT (`raw PREROUTING -i glabs-api0 -d <agent_api ip> -p tcp --dport <port> -j ACCEPT`) **above** it. Every reaper tick
   checks the rules. `/v1/health` reports `firewall: ok|missing`, and creates are refused (`firewall_missing` 503) while it is missing.
6. **Caller origin is a header.** Every request carries `X-Labs-Origin: owner|autopilot|probe`, and a missing header counts as `autopilot`
   (fail closed). `last_owner_use_at` is refreshed only by lab-scoped calls with `owner`. `owner_only`, the destroy guard and pre-emption use
   the header, not the lab's labels. (The create body's `origin` field is removed; the header is stored as the lab's origin label.)
7. **Kept machines: members and data.** `app.member` is required. `/keep` **recreates every member** from the spec with
   `restart: unless-stopped` and a per-member data volume `ghost-labdata-<lineage>-<app>-<member>` at `/data`. Anything outside `/data` is
   lost, so deploy after keep. Kept `pg:18` uses `PGDATA=/data/pg`, and kept `mysql:8` uses `datadir=/data/mysql`.
8. **Switch sequencing.**
   - `data: shared` (default): v2's lab mounts **nothing** until `switch`. A deploy on an unswitched lab installs but does not start the app (state `staged`).
   - `switch` takes both labs' locks in id order, then: snapshot → stop v1's app → recreate v2's app member with the shared volume → start → probe → move the lease.
   - Probe failure → automatic reverse, and the result says so.
   - `data: fresh` names the volume `…-<lab_id>` (owned by that version only).
   - The `data_seed` marker is `/data/.ghost-seeded`, per volume.
9. **v1 leftovers are void:** v1 §6 entirely (proxy env, SocksPort/HTTPTunnelPort, the `-pub` network, the `ports: not_supported` fallback)
   and v1 §11 test 4 (in a tor lab `--noproxy` now **succeeds through Tor**; the test asserts `IsTor:true` with and without `--noproxy`). No
   proxy env is set. Shipped templates: `blank-dev`, `web`, `mysql-single` = `tor`; `pg-single`, `patroni-3` = `none` (no internet needed;
   `params` may opt into tor).
10. **Route re-application** happens on Docker `start` **and** `network:connect` events, **synchronously inside create/start/rejoin**, and
    in a sweep every reaper tick (the events listener may miss events during a dockerd restart). Readiness and app start wait until
    `ip route show default` points at the gateway. The window before the route is fail-closed by construction: the internal network has
    no default route, and the host is behind INPUT DROP.
11. **Agent-api proxy precision.**
    - It forwards only `Content-Type, Content-Length, Accept, X-Ghost-Service-Token`, sets `Host` to the upstream, and adds no `X-Forwarded-*`.
    - It rejects any path containing `..`, `//`, `%2e`, `%2f` or `%5c` (case-insensitive) or failing `^/api/game/[A-Za-z0-9/_-]*$`, follows no
      redirects, caps request and response bodies at 64 KB / 1 MB, and has a 30 s timeout.
    - The sidecar has `ip_forward=0`, network alias `agent-api`, and sits on the lab network + `ghost-labs-api`.
    - `agent_api_url` must be an `http://<100.64/10 IP literal>:<port>`, and it is a field of **`/keep`** (and of create).
    - The service token goes in `secret_values`.
12. **Tests added to §10:**
    - lab A → lab B unreachable (read data, through the shared gateway and directly);
    - from the gateway's netns: LAN, host sshd and tailnet fail;
    - after `docker restart ghost-labs-tor`, a member is still Tor-only and clearnet still fails;
    - an `egress: none` lab: no external DNS, no TCP;
    - `resolv.conf` has no `ts.net`;
    - "another tailnet host unreachable" runs **from the sidecar's netns**;
    - the port is unchanged after rehearse and after switch;
    - `data: fresh` isolation; rollback with `restore_snapshot: true`;
    - after `ghost-labs-fw` rules are deleted: health `firewall: missing` and creates refused;
    - reboot as a scripted `make itest-reboot` (a `pre` phase records state, the operator reboots, a `post` phase asserts routes,
      forwarders, apps, the firewall and data).

### MED
13. **Backups:** `backup: {cmd, restore_cmd}`. Volumes are read with `get_archive` from the member and restored with `put_archive`
    (no host bind mounts), then `restore_cmd` runs. No hook → stop app → archive → start app. The manager's sqlite
    (config, leases, secrets) is backed up daily as well, next to the data backups. (§4UL round 3: postgres members
    are copied stopped even with a hook, and a restore that put back postgres files does not run `restore_cmd`.)
14. **App control:**
    - `POST /v1/labs/{id}/app/{start|stop|restart}` (owner origin for kept labs); `start` is the way out of `crashloop`.
    - Lab detail gains `app: {name, member, state: staged|starting|up|restarting|crashloop|stopped, port, url, restarts, last_probe_at, last_error}`.
    - Restart after **3 consecutive failed probes**, with a start grace of 30 s and backoff 5/15/45… s capped at 300 s.
    - A kept member's `start` event (reboot, Docker restart) starts its app **without counting a crash**.
    - Secrets go into the app's exec env, plus a 0600 `/run/ghost/secrets.env` for cron, **not** into the container config.
15. **`python-app` provides `/opt/venv` on PATH** (PEP 668: no system pip on trixie). Deploy's pip deps go there, and apt deps install as root.
16. **Deploy archive:** `archive_b64` only (≤ `file_max_bytes`; larger → `lab put` the tarball first, then `archive_member_path`, a path
    **inside `app.member`**). Every archive is kept under `/var/lib/ghost-labs/archives/<lineage>/<app>/<ts>.tgz` (last 5) for `rehearse`/rollback.
    Extraction happens inside the member only. Each `data_seed` path in `/app` is replaced by a **symlink into `/data`**, so apps that keep
    data next to their code (jiu-jitsu `data.db`, chess-coach `saves/`, `game_state.json`) keep it across redeploys.
17. **Export (Phase E):** `export {passphrase}` → AES-256-GCM with a scrypt-derived key. `import {path, passphrase}`.
18. **Kept lifecycle:**
    - The `from_lab` of a switch gets `retire_at` (now + 24 h), which the reaper honours despite the kept exemption (reason `retired`).
    - The rehearsal twin counts against the **kept** pool.
    - A single `DELETE` of a kept lab requires `owner` + `?include_kept=1`.
    - Reconcile adopts any lab labelled `lifetime=kept` (`adopted_unreadable: true` when sqlite lacks it). Missing secrets → app `stopped`,
      `last_error: secrets_lost`.
19. **No SSRF:** `/http` and `/screenshot` follow no redirects and connect only to the named member's IP:port (`/screenshot` takes `member`
    + `port` + `path`, not a free URL). The screenshot helper is created exactly like a member (dns, route, INPUT DROP) and is disconnected
    and removed after use.
20. **Redaction** covers the raw, base64, URL-encoded, hex and JSON-escaped forms of every secret, applied **before truncation** and across
    line joins. It is documented as transcript hygiene, **not** a security boundary.
21. **Lifecycle details:**
    - The TTL class comes from the owner: `project:` → project, `conversation:` → adhoc, origin probe → probe.
    - v1 `default_ttl_idle_s`/`max_lifetime_s` are superseded, and `extend` is capped at the class `max_s`.
    - Ad-hoc `members` take `{name, image, mem_mb, cpus, ports?, cap_add?}`, plus lab-level `egress` and `readiness` (default `true`).
    - Pre-emption also applies to `lab_limit`, and MemAvailable is re-checked after evicting.
    - `PUT /v1/labs/{id}/secrets {secret_values}` (owner) rotates secrets and restarts the app.
    - Infrastructure objects (Tor data volume, `ghost-labs-gw`, `ghost-labs-api`, gateway/sidecar containers) carry
      `ghost.labs.role=infra` and are exempt from the reconcile sweep.

### LOW
22. **Codes added:** `unknown_backup` 404, `no_rollback` 409, `ports_exhausted` 409, `already_kept` 409, `firewall_missing` 503. Destroyed
    reasons add `evicted, retired, owner_delete, reconcile`. Lab states add `staged, switching, rehearsing, stopped`.
23. **Small items:**
    - The gateway gets `mem_limit 256m` and `oom_score_adj -500`.
    - `ghost-lab/mysql:8` builds `FROM mysql:8.0` (the official multi-arch image, pulled at build time only) + tini + the lab entrypoint, and
      is not Debian-packaged.
    - Tor bootstrap is read via `ControlPort` (cookie auth) `GETINFO status/bootstrap-phase`.
    - `destroy --all` is an error that points at `--all-disposable`.

## 12. App exposure (UNIFIED §15, operator X1–X3 YES, 2026-11-03), overrides §4 and §11.2 where they differ
- Forwarders bind **`127.0.0.1:<port>`** (plus the LAN IP only for apps with `lan: true`), never the tailnet IP directly.
- The tailnet side: the service runs `tailscale serve --bg --https=<port> http://127.0.0.1:<port>` per published port, and `… off` on
  stop/destroy/lease move. It is reconciled at service start against `tailscale serve status --json`, and only ports in
  `published_port_range` (plus 443 for the Apps page) are ever touched.
- URLs are returned as `https://<Self.DNSName from tailscale status --json>:<port>/`, plus `lan_url` when LAN is on.
- **Per-app names (X2):** when config `ts_services: true`, a kept app's lease also gets `tailscale serve --service=svc:<app> --https=443
  http://127.0.0.1:<port>`, and the URL becomes `https://<app>.<tailnet>.ts.net/` (the port URL stays as a fallback). **ON by default** (day-1 test passed 2026-11-04, UNIFIED §15.5). Wait for the name to answer (retry ≤ 2 min) before reporting it.
- `/keep` takes `lan: false` (default) | `true`. LAN binds use the host's primary LAN IPv4 (detected at runtime) and are listed on the
  Apps page with a "no login: home network only" note.
- `GET /` on 443: the read-only Apps page (no auth on the page itself, because it is tailnet-only via serve; it shows names, states and links,
  never secrets or ids beyond lab ids). `PUT /v1/previews` (agent, X-Labs-Key) replaces the preview list shown there.
- Tests: an app reachable over HTTPS on the tailnet name with a valid certificate, and not on the tailnet IP's plain port; LAN off → not
  reachable on the LAN IP; LAN on → reachable; serve entries removed on destroy; the Apps page lists apps and previews.


## 13. §4UL (2026-11-12): templates, owner composition, build step, in-place deploy

- `POST /v1/labs` adds `overrides {mem_mb, ports, cap_add, cpus, egress}` (a plain value or `{member: value}`) on a template; the
  members are validated again with the template rules. `members` and `overrides` are **owner origin only** (`owner_only`).
- A template may carry `app {member, port, cmd, health, build, service}`; `keep` fills what the caller left out; `/v1/templates` lists it.
- Kept app fields added: `build` (null = auto, "" = none, else a command), `subdir` (the agent ships `<folder>/<subdir>`), `service`.
- Deploy: the build runs in `<workdir>.new` before the swap; `requirements.build` overrides it per deploy. New error `build_failed` (422).
- Deploy answer adds `rolled_back` (+ `failed_probe`, `rollback_error`) when the previous archive was redeployed, and `still_starting`.
  `probe.build_log` carries the build's tail.
- `PUT /v1/labs/{id}/app/config {cmd?, health?, port?, workdir?, build?, subdir?}` → `{lab_id, app, probe, changed}`.
- Images added: `ghost-lab/static:1`, `ghost-lab/node-app:1`.
