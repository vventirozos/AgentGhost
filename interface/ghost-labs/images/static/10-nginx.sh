#!/bin/bash
# a disposable static-site lab serves /app at once; a kept one's app cmd watches this same nginx
mkdir -p /app
[ -n "$(ls -A /app 2>/dev/null)" ] || echo "<h1>static site: deploy a folder</h1>" > /app/index.html
pgrep -x nginx >/dev/null || nginx
