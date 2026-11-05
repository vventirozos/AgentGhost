import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_configure(config):
    for m in ("integration", "reboot_pre", "reboot_post"):
        config.addinivalue_line("markers", f"{m}: needs the real labs host")
