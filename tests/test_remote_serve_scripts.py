"""Shape/regression guard for the tailscale-serve host helpers
(serve-remote.sh / unserve-remote.sh, 2026-07-12).

These scripts put a PUBLISHED sandbox service on the operator's tailnet — the
host half of "the agent hosts something and I reach it remotely". They live at
the ops-script location (``/Users/vasilis/Data/AI/bin/`` by default, outside
this repo), so the test SKIPS when they're absent (a fresh checkout / CI) and
validates the invariants on the host where they're deployed.

Invariants that matter (each maps to a bug we'd regress):
  * background serve with a 1:1 HTTPS<->local port map (`--bg --https=<p> <p>`),
  * robust tailscale CLI discovery incl. the macOS GUI app bundle (the CLI is
    NOT on PATH there — the whole feature silently no-ops without this),
  * targeted teardown (`--https=<p> off`) plus a full `reset` for `all`,
  * strict-mode bash + a usage guard so a bare call can't serve a wrong port.
"""

import os
import stat

import pytest

from ghost_agent.sandbox.services import (
    REMOTE_SERVE_SCRIPT, REMOTE_UNSERVE_SCRIPT,
)


def _read_or_skip(path):
    if not os.path.exists(path):
        pytest.skip(f"host helper not deployed at {path} (ops-script location)")
    with open(path, "r") as f:
        return f.read()


class TestServeRemoteScript:
    def test_exists_and_executable(self):
        if not os.path.exists(REMOTE_SERVE_SCRIPT):
            pytest.skip("serve-remote.sh not deployed on this host")
        mode = os.stat(REMOTE_SERVE_SCRIPT).st_mode
        assert mode & stat.S_IXUSR, "serve-remote.sh must be executable"

    def test_backgrounds_with_1to1_https_map(self):
        src = _read_or_skip(REMOTE_SERVE_SCRIPT)
        # `tailscale serve --bg --https=<tailnet_port> <local_port>`
        assert "serve --bg --https=" in src
        # Default tailnet port == local port (1:1) when arg 2 is omitted.
        assert 'TAILNET_PORT="${2:-${LOCAL_PORT}}"' in src

    def test_discovers_cli_including_mac_app_bundle(self):
        src = _read_or_skip(REMOTE_SERVE_SCRIPT)
        assert "command -v tailscale" in src
        assert "/Applications/Tailscale.app/Contents/MacOS/Tailscale" in src

    def test_strict_mode_and_usage_guard(self):
        src = _read_or_skip(REMOTE_SERVE_SCRIPT)
        assert "set -euo pipefail" in src
        # A bare invocation must refuse (exit non-zero) rather than serve.
        assert "exit 2" in src

    def test_points_at_teardown(self):
        src = _read_or_skip(REMOTE_SERVE_SCRIPT)
        assert "unserve-remote.sh" in src


class TestUnserveRemoteScript:
    def test_targeted_off_and_full_reset(self):
        src = _read_or_skip(REMOTE_UNSERVE_SCRIPT)
        # Targeted removal of one mapping ...
        assert 'serve --https="${ARG}" off' in src
        # ... and `all`/no-arg clears everything.
        assert "serve reset" in src

    def test_discovers_cli_including_mac_app_bundle(self):
        src = _read_or_skip(REMOTE_UNSERVE_SCRIPT)
        assert "/Applications/Tailscale.app/Contents/MacOS/Tailscale" in src
