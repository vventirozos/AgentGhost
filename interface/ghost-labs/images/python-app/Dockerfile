FROM ghost-lab/base:1
# PEP 668: no system pip on trixie; apps install into /opt/venv (contract v2 §11.15).
RUN python3 -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir flask gunicorn requests
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv
# exec runs `bash -lc`; /etc/profile resets PATH, so put the venv back for login shells too
RUN echo 'export PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv' > /etc/profile.d/10-venv.sh
