FROM ghost-lab/base:1
RUN apt-get update && apt-get install -y --no-install-recommends chromium fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
CMD ["chromium", "--version"]
