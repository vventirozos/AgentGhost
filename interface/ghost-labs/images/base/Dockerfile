FROM debian:trixie-slim
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash sudo ca-certificates curl wget git openssh-client rsync vim nano less jq htop procps psmisc \
      iproute2 iputils-ping dnsutils netcat-openbsd tcpdump strace lsof build-essential pkg-config \
      python3 python3-venv python3-pip python3-dev sqlite3 make unzip xz-utils cron iptables \
    && rm -rf /var/lib/apt/lists/*
COPY images/base/lab-entrypoint /usr/local/bin/lab-entrypoint
RUN chmod 0755 /usr/local/bin/lab-entrypoint && mkdir -p /etc/lab/start.d /var/log/lab/jobs /data /app
WORKDIR /root
CMD ["/usr/local/bin/lab-entrypoint"]
