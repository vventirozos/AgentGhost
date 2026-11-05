#!/bin/bash
# Initialise on first start (PGDATA; kept labs use /data/pg), then start.
set -u
PGDATA=${PGDATA:-/var/lib/postgresql/data}
BIN=$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -1)
mkdir -p "$PGDATA" && chown -R postgres:postgres "$(dirname "$PGDATA")" "$PGDATA" && chmod 700 "$PGDATA"
if [ ! -s "$PGDATA/PG_VERSION" ]; then
  echo lab > /tmp/pwfile
  sudo -u postgres "$BIN/initdb" -D "$PGDATA" --auth-local=trust --auth-host=scram-sha-256 --pwfile=/tmp/pwfile
  rm -f /tmp/pwfile
  echo "listen_addresses = '*'" >> "$PGDATA/postgresql.conf"
  echo "host all all 0.0.0.0/0 scram-sha-256" >> "$PGDATA/pg_hba.conf"
fi
mkdir -p /var/run/postgresql && chown postgres:postgres /var/run/postgresql
# cwd must be readable by postgres (the entrypoint runs in /root; Patroni's launcher dies on it)
cd /var/lib/postgresql
exec sudo -H -u postgres "$BIN/postgres" -D "$PGDATA"
