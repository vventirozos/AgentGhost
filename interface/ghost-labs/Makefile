PY ?= /opt/ghost-labs/venv/bin/python
test:
	$(PY) -m pytest -q tests/unit
itest:
	$(PY) -m pytest -q -m integration tests/integration
itest-reboot-pre:
	$(PY) -m pytest -q -m reboot_pre tests/integration/test_reboot.py
itest-reboot-post:
	$(PY) -m pytest -q -m reboot_post tests/integration/test_reboot.py
.PHONY: test itest itest-reboot-pre itest-reboot-post
