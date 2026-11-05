#!/bin/bash
export MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD:-lab}
DATADIR=${MYSQL_DATADIR:-/var/lib/mysql}
exec docker-entrypoint.sh mysqld --datadir="$DATADIR" --bind-address=0.0.0.0
