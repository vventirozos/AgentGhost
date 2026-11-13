#!/bin/bash
# §4UL round 6: a restore of /data that did not complete leaves this marker (written by ghost-labs before /data is
# cleared, removed only by a restore that completed). While it exists mysqld is neither initialised nor started —
# the official entrypoint would initialise an EMPTY datadir and serve it. The container stays up without mysqld.
MARK=${GHOST_RESTORE_MARKER:-/data/.ghost-restore-incomplete}   # the variable: tests only
if [ -e "$MARK" ]; then
  echo "$(date -Is) mysqld NOT started: a restore of /data did not complete ($MARK). Restore a good backup, or" \
       "start the app with reinit to accept /data as it is." >&2
  exit 0
fi
export MYSQL_ROOT_PASSWORD=${MYSQL_ROOT_PASSWORD:-lab}
DATADIR=${MYSQL_DATADIR:-/var/lib/mysql}
exec docker-entrypoint.sh mysqld --datadir="$DATADIR" --bind-address=0.0.0.0
