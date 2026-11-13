FROM ghost-lab/base:1
# §4UL: the static-site template's box — nginx serves /app (the folder a kept deploy replaces) on 8080,
# with no directory listing and no dotfiles (.git, .env, …)
RUN apt-get update && apt-get install -y --no-install-recommends nginx && rm -rf /var/lib/apt/lists/* \
    && rm -f /etc/nginx/sites-enabled/default
COPY images/static/default.conf /etc/nginx/conf.d/default.conf
COPY images/static/10-nginx.sh /etc/lab/start.d/10-nginx.sh
