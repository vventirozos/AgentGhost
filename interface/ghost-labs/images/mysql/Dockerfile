# Official multi-arch image (pulled at build time only, D5) + the lab entrypoint.
FROM mysql:8.0
# setsid/pgrep for the manager's exec and jobs (Oracle Linux slim lacks them)
RUN microdnf install -y util-linux procps-ng && microdnf clean all
COPY images/base/lab-entrypoint /usr/local/bin/lab-entrypoint
COPY images/mysql/10-mysql.sh /etc/lab/start.d/10-mysql.sh
RUN chmod 0755 /usr/local/bin/lab-entrypoint && mkdir -p /var/log/lab/jobs /data /app
ENTRYPOINT []
CMD ["/usr/local/bin/lab-entrypoint"]
