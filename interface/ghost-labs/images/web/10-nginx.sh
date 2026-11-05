#!/bin/bash
mkdir -p /srv/www
[ -e /srv/www/index.html ] || echo "<h1>ghost lab web</h1>" > /srv/www/index.html
exec nginx -g 'daemon off;'
