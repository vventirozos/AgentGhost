FROM ghost-lab/base:1
RUN apt-get update && apt-get install -y --no-install-recommends nginx && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default
COPY images/web/default.conf /etc/nginx/conf.d/default.conf
COPY images/web/10-nginx.sh /etc/lab/start.d/10-nginx.sh
