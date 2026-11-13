#!/bin/bash
# Initialise on first start (PGDATA; kept labs use /data/pg), then start.
set -u
PGDATA=${PGDATA:-/var/lib/postgresql/data}
BIN=$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -1)
# §4UL round 5: a restore of /data that did not complete leaves this marker (written by ghost-labs BEFORE /data is
# cleared, removed only by a restore that completed). While it exists postgres is neither initialised nor started —
# on any container start (reboot, docker restart, OOM, a recreate): an initdb here served an EMPTY database on the
# app's port. The container stays up without postgres. The restore itself starts postgres with
# GHOST_PG_RESTORING=1: it may start the restored files, but never initdb.
MARK=${GHOST_RESTORE_MARKER:-/data/.ghost-restore-incomplete}   # the variable: tests only
# round 6: checked again right before initdb and before postgres starts — a restore may begin while this script
# sits in `chown -R` (ghost-labs also kills a running copy before it writes the marker)
held() {
  if [ -e "$MARK" ] && [ "${GHOST_PG_RESTORING:-}" != 1 ]; then
    echo "$(date -Is) postgres NOT started ($1): a restore of /data did not complete ($MARK). Restore a good" \
         "backup, or start the app with reinit to accept /data as it is." >&2
    return 0
  fi
  return 1
}
held "at start" && exit 0
mkdir -p "$PGDATA" && chown -R postgres:postgres "$(dirname "$PGDATA")" "$PGDATA" && chmod 700 "$PGDATA"
if [ ! -s "$PGDATA/PG_VERSION" ] && [ "${GHOST_PG_RESTORING:-}" = 1 ]; then
  echo "$(date -Is) postgres NOT started: the restored /data has no cluster ($PGDATA/PG_VERSION) — no initdb" \
       "during a restore." >&2
  exit 1
fi
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  held "before initdb" && exit 0
  echo lab > /tmp/pwfile
  sudo -u postgres "$BIN/initdb" -D "$PGDATA" --auth-local=trust --auth-host=scram-sha-256 --pwfile=/tmp/pwfile
  rm -f /tmp/pwfile
  echo "listen_addresses = '*'" >> "$PGDATA/postgresql.conf"
  echo "host all all 0.0.0.0/0 scram-sha-256" >> "$PGDATA/pg_hba.conf"
fi
mkdir -p /var/run/postgresql && chown postgres:postgres /var/run/postgresql
# cwd must be readable by postgres (the entrypoint runs in /root; Patroni's launcher dies on it)
cd /var/lib/postgresql
held "before postgres" && exit 0
exec sudo -H -u postgres "$BIN/postgres" -D "$PGDATA"
