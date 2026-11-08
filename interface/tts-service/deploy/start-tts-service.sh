#!/bin/bash
# launchd entry point. Binds the tailnet address only, and waits for it at boot: launchd can start this before
# Tailscale is up, and falling back to another interface would put reply text on the LAN.
# Only a utun interface counts: a 100.64/10 address on en0 would be a carrier's NAT, not the tailnet.
set -u
S="$HOME/Data/AI/tts-service"
ip=""
for _ in $(seq 1 60); do
  ip=$(/sbin/ifconfig | /usr/bin/awk '
    /^[a-z0-9]+:/ { iface = $1 }
    iface ~ /^utun/ && /inet 100\./ { split($2, o, "."); if (o[2] >= 64 && o[2] <= 127) { print $2; exit } }')
  [ -n "$ip" ] && break
  sleep 5
done
if [ -z "$ip" ]; then
  echo "no tailnet address after 300 s; launchd will start this script again" >&2
  exit 1
fi
export HF_HOME="$S/hf"
export HF_HUB_OFFLINE=1
cd "$S/app" || exit 1
exec "$S/venv/bin/python" -u tts_service.py --host "$ip" --voice Maeve
