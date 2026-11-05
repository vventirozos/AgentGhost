FROM ghost-lab/base:1
RUN apt-get update && apt-get install -y --no-install-recommends postgresql-common gnupg lsb-release \
    && /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y \
    && apt-get install -y --no-install-recommends postgresql-18 postgresql-client-18 postgresql-contrib \
    && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/lib/postgresql/18/bin:$PATH
# exec runs `bash -lc`; /etc/profile resets PATH
RUN echo 'export PATH=/usr/lib/postgresql/18/bin:$PATH' > /etc/profile.d/10-pg.sh
COPY images/pg/10-postgres.sh /etc/lab/start.d/10-postgres.sh
