# Ghost Agent sandbox image — everything-preinstalled form.
#
# Build:
#   docker build -f sandbox/Dockerfile -t ghost-agent-base:latest .
#
# Why bake instead of runtime-provision?
#   The previous flow lazily installed the deep-learning stack + Playwright
#   Chromium at first container start and committed the result as
#   `ghost-agent-base:latest`. That worked, but had two sharp edges:
#
#     1. If the install sequence had a bug (e.g. `playwright install
#        chromium` without `--with-deps`, as the image of record did
#        for a while), the committed cache was silently wrong. Every
#        subsequent boot trusted `/root/.supercharged` and skipped the
#        now-correct install flow — so the agent kept discovering at
#        *self-play time* that Chromium system deps were missing, burned
#        ~100 s rediscovering, and still couldn't run the browser.
#     2. The first-boot wait was 60 s for pip + ~120 s for Chromium,
#        paid by the first user of every fresh install.
#
#   A Dockerfile gets both. It's reproducible, diff-reviewable, and the
#   runtime sandbox marker bump (`.supercharged.v2`) guarantees anyone
#   still running the old cached image gets a fresh install on next
#   launch.
#
# Size budget: ~2 GB. Chromium alone is ~350 MB; the deep-learning
# stack adds another ~1 GB. This is the tradeoff for zero-wait sandbox
# startup on a privacy-first agent that can't pull images at runtime.

FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright \
    # Every telemetry opt-out Ghost requires. Kept in sync with
    # `ghost_agent._env._REQUIRED_FLAGS` — if you add a flag there,
    # add it here so sandboxed processes inherit the same posture.
    ANONYMIZED_TELEMETRY=False \
    POSTHOG_DISABLED=1 \
    TELEMETRY_IMPL=none \
    CHROMA_TELEMETRY_IMPL=none \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DISABLE_VERSION_CHECK=1

# --- System packages ---
# sudo          : tool execution needs passwordless sudo for apt/root ops
# nodejs/npm    : for JS sandbox targets
# g++           : compiled-code targets (pandas wheels fallback, C extensions)
# curl/wget/git : fetch fixtures inside tasks
# procps        : ps/kill for sandbox observability
# postgresql-* / libpq-dev : DBA-agent postgres client
# tor           : optional Tor-proxied network (the agent runs this
#                 as the debian-tor user when --tor-proxy is configured)
# ripgrep       : fast code search
# sqlite3       : lightweight data-analysis tasks
# iproute2      : `ss` — socket/port inspection (what's listening)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        sudo coreutils nodejs npm g++ curl wget git procps \
        postgresql-client libpq-dev tor ripgrep sqlite3 iproute2 \
 && rm -rf /var/lib/apt/lists/*

# Passwordless sudo — the sandbox runs as root by default, but the
# self-play harness sometimes invokes sudo explicitly inside solutions.
RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# --- Python deep-learning / data stack ---
# Mirrors the runtime install_cmd in sandbox/docker.py. Kept here so
# the image is a drop-in replacement that skips the first-boot wait.
RUN pip install --no-cache-dir \
        numpy pandas scipy matplotlib seaborn plotly \
        scikit-learn yfinance beautifulsoup4 networkx requests \
        pylint black mypy bandit dill ipykernel jupyter_client \
        pytest pytest-asyncio \
        psycopg2-binary asyncpg sqlalchemy tabulate sqlglot \
        playwright html2text lxml pysocks \
        flask python-chess

# PyTorch — CPU wheel only (the default GPU wheels pull ~2 GB of unusable
# CUDA). Preinstalled so "train a model" projects don't `pip install torch`
# mid-task and trip the execute timeout. Mirrors the runtime install in
# sandbox/docker.py.
RUN pip install --no-cache-dir torch \
        --index-url https://download.pytorch.org/whl/cpu

# --- Playwright browsers + system deps ---
# `--with-deps` installs the libnss3 / libatk / libdrm / etc. that
# Chromium's headless_shell links against. Running this step inside
# the image (instead of at runtime) means:
#   (a) the image is a complete, reproducible artefact — no "install
#       succeeded, committed, ran fine, next boot is broken" drift.
#   (b) the self-play harness doesn't burn turns on discovering and
#       re-installing Chromium at task execution time.
RUN python3 -m playwright install --with-deps chromium

# Sentinel: when the runtime code in sandbox/docker.py finds this
# marker, it skips every provisioning step. Versioned so older
# cached images (v1 or missing) trigger a fresh provision. Bump this
# tag (and the corresponding string in docker.py) whenever this
# Dockerfile's install surface changes in a way that prior images
# can't be trusted to match.
RUN touch /root/.supercharged.v4

WORKDIR /workspace
CMD ["sleep", "infinity"]
