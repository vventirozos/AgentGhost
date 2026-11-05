FROM debian:trixie-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends tor iptables iproute2 ca-certificates \
      netcat-openbsd gawk && rm -rf /var/lib/apt/lists/*
COPY images/tor-gw/torrc /etc/tor/torrc
COPY images/tor-gw/gw-entrypoint /usr/local/bin/gw-entrypoint
RUN chmod 0755 /usr/local/bin/gw-entrypoint
CMD ["/usr/local/bin/gw-entrypoint"]
