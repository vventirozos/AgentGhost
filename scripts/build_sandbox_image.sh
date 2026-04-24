#!/usr/bin/env bash
# Build the pre-provisioned sandbox image.
#
# Run this once after install / after pulling a new Ghost version.
# The agent's runtime sandbox wrapper will pick the freshly-built
# `ghost-agent-base:latest` on its next `ensure_running`.
#
# Usage:
#   scripts/build_sandbox_image.sh
#
# Notes:
#   * Takes ~5 min on a warm docker cache (pip + Chromium are the
#     long poles). First run from a cold cache pulls ~2 GB.
#   * If a container named `ghost_sandbox` is currently running
#     against the OLD image, restart the agent after the build so
#     it picks up the new one.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not on PATH" >&2
    exit 1
fi

TAG="ghost-agent-base:latest"

echo "==> Building ${TAG} from sandbox/Dockerfile"
docker build \
    --file sandbox/Dockerfile \
    --tag "${TAG}" \
    --progress=plain \
    .

echo ""
echo "==> Build complete. Verifying Chromium install..."
docker run --rm "${TAG}" \
    python3 -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
    page = browser.new_page()
    page.set_content('<h1>ok</h1>')
    title = page.content()
    browser.close()
    assert '<h1>ok</h1>' in title, 'chromium smoke test failed'
    print('chromium smoke test ok')
"

echo ""
echo "==> ${TAG} ready. Restart the agent to pick it up."
