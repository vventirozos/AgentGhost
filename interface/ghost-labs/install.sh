#!/bin/bash
# ghost-labs installer: parameterless, idempotent, any arch (contract v2 §1).
# Run as root from the unpacked tree at /opt/ghost-labs (setup-time internet is allowed: D5).
set -euo pipefail
OPT=/opt/ghost-labs ETC=/etc/ghost-labs VAR=/var/lib/ghost-labs
[ "$(id -u)" = 0 ] || { echo "run as root" >&2; exit 1; }
[ "$(cd "$(dirname "$0")" && pwd)" = "$OPT" ] || { echo "run from $OPT" >&2; exit 1; }
command -v docker >/dev/null || { echo "Docker is required" >&2; exit 1; }
command -v tailscale >/dev/null || { echo "Tailscale is required" >&2; exit 1; }

echo "== packages"
need=""
for p in socat python3-venv iptables util-linux; do dpkg -s "$p" >/dev/null 2>&1 || need="$need $p"; done
[ -z "$need" ] || { apt-get update -q && apt-get install -y -q $need; }

echo "== venv"
[ -x "$OPT/venv/bin/python" ] || python3 -m venv "$OPT/venv"
"$OPT/venv/bin/pip" install -q --upgrade pip
"$OPT/venv/bin/pip" install -q -r "$OPT/requirements.txt"

echo "== dirs, key, host id"
install -d -m 0755 "$ETC" "$ETC/templates"
install -d -m 0700 "$VAR"
chmod 0700 "$VAR"; chmod -R go-rwx "$VAR"
install -d -m 0700 "$ETC/fwd" "$VAR/backups" "$VAR/archives"
[ -s "$ETC/api_key" ] || (umask 077; head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$ETC/api_key")
chmod 0600 "$ETC/api_key"
[ -s "$ETC/host_id" ] || python3 -c 'import uuid;print(uuid.uuid4())' > "$ETC/host_id"
# shipped templates are read from $OPT/templates (the code's tree); $ETC/templates holds only the operator's own.
# A copy of a shipped one left there by an older install is ignored by name (the service logs it) — remove them:
for f in "$OPT"/templates/*.yaml; do b="$ETC/templates/$(basename "$f")"; [ ! -e "$b" ] || ! cmp -s "$f" "$b" || rm -f "$b"; done
[ -e "$ETC/config.yaml" ] || printf '# ghost-labs config; every key optional (see README)\n{}\n' > "$ETC/config.yaml"

POOL=$("$OPT/venv/bin/python" -c 'import sys;sys.path.insert(0,"'"$OPT"'");from ghost_labs import config as C;c=C.load();print(c.address_pool, c.gw_subnet, c.api_subnet)')
read -r POOL_NET GW_NET API_NET <<<"$POOL"
printf 'POOL=%s\nGW_SUBNET=%s\nAPI_SUBNET=%s\n' "$POOL_NET" "$GW_NET" "$API_NET" > "$ETC/fw.env"

echo "== docker daemon.json"
changed=$("$OPT/venv/bin/python" - "$POOL_NET" <<'PY'
import json, os, sys
p = "/etc/docker/daemon.json"
cur = json.load(open(p)) if os.path.exists(p) and os.path.getsize(p) else {}
want = dict(cur)
want["default-address-pools"] = [{"base": sys.argv[1], "size": 24}]  # fence: every Docker network sits behind the INPUT DROP
want["log-opts"] = {**cur.get("log-opts", {}), "max-size": "10m", "max-file": "3"}
if want != cur:
    tmp = p + ".tmp"
    json.dump(want, open(tmp, "w"), indent=2)
    os.replace(tmp, p)
    print("yes")
else:
    print("no")
PY
)

echo "== systemd units"
install -m 0644 "$OPT"/systemd/*.service /etc/systemd/system/
ln -sf "$OPT/bin/ghost-labs" /usr/local/bin/ghost-labs
systemctl daemon-reload
systemctl enable ghost-labs-fw.service
systemctl restart ghost-labs-fw.service
if [ "$changed" = yes ]; then echo "docker config changed: restarting docker"; systemctl restart docker; fi

echo "== images (built for $(uname -m))"
GHOST_LABS_OPT_DIR=$OPT "$OPT/bin/ghost-labs" build-images

echo "== service"
systemctl enable ghost-labs.service
systemctl restart ghost-labs.service
for i in $(seq 1 30); do
  systemctl is-active -q ghost-labs.service && "$OPT/bin/ghost-labs" status >/dev/null 2>&1 && break
  sleep 2
done
systemctl is-active ghost-labs.service
"$OPT/bin/ghost-labs" url
