FROM ghost-lab/base:1
# §4UL: Node.js + npm from Debian trixie. A kept deploy runs `npm ci` (or `npm install`) and
# `npm run build --if-present` in the new folder; npm reaches the registry through the lab's Tor egress
# (transparent proxy, no npm proxy settings needed). node_modules lives in the app folder.
RUN apt-get update && apt-get install -y --no-install-recommends nodejs npm && rm -rf /var/lib/apt/lists/* \
    && npm config set --global fund false && npm config set --global update-notifier false
