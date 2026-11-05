FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
COPY images/agentapi/proxy.py /opt/proxy.py
USER nobody
CMD ["python3", "/opt/proxy.py"]
