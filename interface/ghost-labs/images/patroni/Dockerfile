FROM ghost-lab/pg:19
# pinned to the versions tested with PostgreSQL 19 (§4VI); bump on purpose and rerun the patroni failover test
RUN apt-get update && apt-get install -y --no-install-recommends etcd-server etcd-client \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/patroni && /opt/patroni/bin/pip install --no-cache-dir 'patroni[etcd3]==4.1.5' 'psycopg2-binary==2.9.13' \
    && ln -s /opt/patroni/bin/patroni /usr/local/bin/patroni && ln -s /opt/patroni/bin/patronictl /usr/local/bin/patronictl \
    && rm -f /etc/lab/start.d/10-postgres.sh
COPY images/patroni/10-patroni.sh /etc/lab/start.d/10-patroni.sh
