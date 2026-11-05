FROM ghost-lab/pg:18
RUN apt-get update && apt-get install -y --no-install-recommends etcd-server etcd-client \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/patroni && /opt/patroni/bin/pip install --no-cache-dir 'patroni[etcd3]' psycopg2-binary \
    && ln -s /opt/patroni/bin/patroni /usr/local/bin/patroni && ln -s /opt/patroni/bin/patronictl /usr/local/bin/patronictl \
    && rm -f /etc/lab/start.d/10-postgres.sh
COPY images/patroni/10-patroni.sh /etc/lab/start.d/10-patroni.sh
