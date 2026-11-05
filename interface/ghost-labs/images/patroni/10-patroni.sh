#!/bin/bash
# etcd (colocated) + Patroni. Members come from PATRONI_MEMBERS="pg1,pg2,pg3";
# this member's name is its hostname.
set -u
ME=$(hostname)
IFS=',' read -ra M <<< "${PATRONI_MEMBERS:?}"
CLUSTER=""
for m in "${M[@]}"; do CLUSTER="${CLUSTER:+$CLUSTER,}$m=http://$m:2380"; done
mkdir -p /var/lib/etcd
nohup etcd --name "$ME" --data-dir /var/lib/etcd \
  --listen-peer-urls http://0.0.0.0:2380 --listen-client-urls http://0.0.0.0:2379 \
  --initial-advertise-peer-urls "http://$ME:2380" --advertise-client-urls "http://$ME:2379" \
  --initial-cluster "$CLUSTER" --initial-cluster-state new --initial-cluster-token ghost-lab \
  >> /var/log/lab/etcd.log 2>&1 &
HOSTS=""
for m in "${M[@]}"; do HOSTS="${HOSTS:+$HOSTS,}$m:2379"; done
BIN=$(ls -d /usr/lib/postgresql/*/bin | sort -V | tail -1)
DATA=${PGDATA:-/var/lib/postgresql/patroni}
mkdir -p "$DATA" && chown -R postgres:postgres "$(dirname "$DATA")" && chmod 700 "$DATA"
cat > /etc/patroni.yml <<YML
scope: ghost-lab
name: $ME
restapi: {listen: 0.0.0.0:8008, connect_address: $ME:8008}
etcd3: {hosts: "$HOSTS"}
bootstrap:
  dcs: {ttl: 30, loop_wait: 10, retry_timeout: 10, maximum_lag_on_failover: 1048576,
        postgresql: {use_pg_rewind: true, parameters: {wal_log_hints: "on"}}}
  initdb: [{encoding: UTF8}, data-checksums]
  pg_hba: ["host replication replicator 0.0.0.0/0 scram-sha-256", "host all all 0.0.0.0/0 scram-sha-256", "local all all trust"]
postgresql:
  listen: 0.0.0.0:5432
  connect_address: $ME:5432
  data_dir: $DATA
  bin_dir: $BIN
  pgpass: /tmp/pgpass0
  authentication:
    superuser: {username: postgres, password: lab}
    replication: {username: replicator, password: lab}
    rewind: {username: rewind_user, password: lab}
YML
chown postgres:postgres /etc/patroni.yml
mkdir -p /var/run/postgresql && chown postgres:postgres /var/run/postgresql
sleep 3
# cwd must be readable by postgres (the entrypoint runs in /root; Patroni's launcher dies on it)
cd /var/lib/postgresql
exec sudo -H -u postgres /opt/patroni/bin/patroni /etc/patroni.yml
